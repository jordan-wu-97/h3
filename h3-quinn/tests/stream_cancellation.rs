//! Integration tests for stream cancellation and reset functionality
//! introduced in commit c32f7d99 ("feat: implement stream cancellation and reset").
//!
//! These tests exercise `RecvStream::recv_reset()`, `SendStream::recv_stopped()`,
//! and the `BidiStream` delegation of both, using real QUIC connections over localhost.
//!
//! A multi-threaded tokio runtime is used so that quinn's I/O tasks can run
//! concurrently with the test logic.
//!
//! **Important**: QUIC streams are not visible to the peer until a STREAM frame
//! is sent on the wire. The helper writes a sentinel byte from the raw-quinn
//! side to materialise the stream.

use std::future::Future;
use std::net::Ipv6Addr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;

use bytes::Bytes;
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

use h3::quic::{self, BidiStream as _, RecvStream as _, SendStream as _};
use h3::quic::StreamErrorIncoming;
use h3_quinn::{self, quinn};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_certs() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    (
        cert.cert.into(),
        PrivateKeyDer::Pkcs8(cert.signing_key.serialize_der().into()),
    )
}

/// Holds both raw quinn connections (for low-level stream ops) and h3-quinn
/// wrappers (for testing trait implementations).
struct Pair {
    client_quinn: quinn::Connection,
    #[allow(dead_code)]
    server_quinn: quinn::Connection,
    #[allow(dead_code)]
    client_h3: h3_quinn::Connection,
    server_h3: h3_quinn::Connection,
}

async fn setup() -> Pair {
    let (cert, key) = build_certs();

    // Server endpoint
    let mut server_crypto = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(vec![cert.clone()], key)
    .unwrap();
    server_crypto.max_early_data_size = u32::MAX;
    server_crypto.alpn_protocols = vec![b"h3".to_vec()];

    let server_config = quinn::ServerConfig::with_crypto(Arc::new(
        QuicServerConfig::try_from(server_crypto).unwrap(),
    ));
    let server_ep = quinn::Endpoint::server(server_config, "[::]:0".parse().unwrap()).unwrap();
    let server_port = server_ep.local_addr().unwrap().port();

    // Client endpoint
    let mut root_store = rustls::RootCertStore::empty();
    root_store.add(cert).unwrap();
    let mut client_crypto = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .unwrap()
    .with_root_certificates(root_store)
    .with_no_client_auth();
    client_crypto.enable_early_data = true;
    client_crypto.alpn_protocols = vec![b"h3".to_vec()];

    let client_config =
        quinn::ClientConfig::new(Arc::new(QuicClientConfig::try_from(client_crypto).unwrap()));
    let mut client_ep = quinn::Endpoint::client("[::]:0".parse().unwrap()).unwrap();
    client_ep.set_default_client_config(client_config);

    let addr = std::net::SocketAddr::from((Ipv6Addr::LOCALHOST, server_port));

    let (client_conn, server_conn) = tokio::join!(
        async { client_ep.connect(addr, "localhost").unwrap().await.unwrap() },
        async { server_ep.accept().await.unwrap().await.unwrap() },
    );

    Pair {
        client_h3: h3_quinn::Connection::new(client_conn.clone()),
        server_h3: h3_quinn::Connection::new(server_conn.clone()),
        client_quinn: client_conn,
        server_quinn: server_conn,
    }
}

/// Client opens a raw bidi stream (writing a sentinel byte to materialise it),
/// server accepts it via h3-quinn `poll_accept_bidi`.
///
/// Returns `(client_quinn_send, client_quinn_recv, server_bidi)`.
///
/// The server-side h3-quinn BidiStream gives access to both:
/// - `RecvStream` (for testing `recv_reset()`)
/// - `SendStream` (for testing `recv_stopped()`)
async fn client_raw_server_h3(
    pair: &mut Pair,
) -> (
    quinn::SendStream,
    quinn::RecvStream,
    h3_quinn::BidiStream<Bytes>,
) {
    let ((cs, cr), server_bidi) = tokio::join!(
        async {
            let (mut s, r) = pair.client_quinn.open_bi().await.unwrap();
            // Write sentinel so the STREAM frame goes on the wire.
            s.write_all(b"\x00").await.unwrap();
            (s, r)
        },
        async {
            futures_util::future::poll_fn(|cx| {
                quic::Connection::<Bytes>::poll_accept_bidi(&mut pair.server_h3, cx)
            })
            .await
            .unwrap()
        },
    );

    (cs, cr, server_bidi)
}

// ---------------------------------------------------------------------------
// RecvStream::recv_reset() tests
// ---------------------------------------------------------------------------

/// Peer resets send side -> recv_reset() yields StreamTerminated { error_code }.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_reset_on_stream_reset() {
    let mut pair = setup().await;
    let (mut client_send, _client_recv, server_bidi) = client_raw_server_h3(&mut pair).await;

    let (_server_send, mut server_recv) = server_bidi.split();

    // Client resets with error code 42.
    client_send.reset(quinn::VarInt::from_u32(42)).unwrap();

    match server_recv.recv_reset().await {
        Some(StreamErrorIncoming::StreamTerminated { error_code }) => {
            assert_eq!(error_code, 42);
        }
        other => panic!("expected StreamTerminated(42), got {:?}", other),
    }
}

/// Peer finishes send side cleanly -> recv_reset() yields None.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_reset_clean_finish_returns_none() {
    let mut pair = setup().await;
    let (mut client_send, _client_recv, server_bidi) = client_raw_server_h3(&mut pair).await;

    let (_server_send, mut server_recv) = server_bidi.split();

    // Client writes data then finishes cleanly.
    client_send.write_all(b"hello").await.unwrap();
    client_send.finish().unwrap();

    // Server drains all data via the trait's poll_data so the stream reaches FIN.
    loop {
        match futures_util::future::poll_fn(|cx| {
            quic::RecvStream::poll_data(&mut server_recv, cx)
        })
        .await
        {
            Ok(None) => break,
            Ok(Some(_)) => continue,
            Err(e) => panic!("unexpected error draining: {:?}", e),
        }
    }

    match server_recv.recv_reset().await {
        None => {} // expected
        other => panic!("expected None for clean finish, got {:?}", other),
    }
}

/// Connection closed by peer -> recv_reset() yields ConnectionErrorIncoming.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_reset_on_connection_close() {
    let mut pair = setup().await;
    let (_client_send, _client_recv, server_bidi) = client_raw_server_h3(&mut pair).await;

    let (_server_send, mut server_recv) = server_bidi.split();

    // Client closes the entire connection.
    pair.client_quinn
        .close(quinn::VarInt::from_u32(7), b"bye");

    match server_recv.recv_reset().await {
        Some(StreamErrorIncoming::ConnectionErrorIncoming { .. }) => {}
        // Timing-dependent: may appear as StreamTerminated during teardown.
        Some(StreamErrorIncoming::StreamTerminated { .. }) => {}
        other => panic!(
            "expected ConnectionErrorIncoming or StreamTerminated, got {:?}",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// SendStream::recv_stopped() tests
// ---------------------------------------------------------------------------

/// Peer sends STOP_SENDING -> recv_stopped() yields StreamTerminated { error_code }.
///
/// Uses the server-side h3-quinn SendStream (obtained by splitting the BidiStream)
/// and the client-side raw recv stream to issue STOP_SENDING.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_stopped_on_stop_sending() {
    let mut pair = setup().await;
    let (_client_send, mut client_recv, server_bidi) = client_raw_server_h3(&mut pair).await;

    let (mut server_send, _server_recv) = server_bidi.split();

    // Client sends STOP_SENDING with error code 99 on server's send direction
    // (= client's recv side of the bidi stream).
    client_recv.stop(quinn::VarInt::from_u32(99)).unwrap();

    match server_send.recv_stopped().await {
        Some(StreamErrorIncoming::StreamTerminated { error_code }) => {
            assert_eq!(error_code, 99);
        }
        other => panic!("expected StreamTerminated(99), got {:?}", other),
    }
}

/// Stream finishes cleanly without STOP_SENDING -> recv_stopped() yields None.
///
/// The server finishes its send side cleanly. The client reads all data.
/// After the stream is complete, recv_stopped() should yield None.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_stopped_clean_finish_returns_none() {
    let mut pair = setup().await;
    let (_client_send, mut client_recv, server_bidi) = client_raw_server_h3(&mut pair).await;

    let (mut server_send, _server_recv) = server_bidi.split();

    // Finish the server's send side cleanly via the trait.
    futures_util::future::poll_fn(|cx| quic::SendStream::<Bytes>::poll_finish(&mut server_send, cx))
        .await
        .unwrap();

    // Client reads to completion.
    while client_recv
        .read_chunk(usize::MAX, true)
        .await
        .unwrap()
        .is_some()
    {}
    drop(client_recv);

    match server_send.recv_stopped().await {
        None => {} // expected
        other => panic!("expected None for clean finish, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// BidiStream delegation tests
// ---------------------------------------------------------------------------

/// BidiStream::recv_reset() delegates to inner RecvStream correctly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bidi_stream_recv_reset_delegates() {
    let mut pair = setup().await;
    let (mut client_send, _client_recv, mut server_bidi) =
        client_raw_server_h3(&mut pair).await;

    // Client resets with error code 7.
    client_send.reset(quinn::VarInt::from_u32(7)).unwrap();

    // Call recv_reset() directly on the BidiStream (tests delegation).
    match quic::RecvStream::recv_reset(&mut server_bidi).await {
        Some(StreamErrorIncoming::StreamTerminated { error_code }) => {
            assert_eq!(error_code, 7);
        }
        other => panic!("expected StreamTerminated(7), got {:?}", other),
    }
}

/// BidiStream::recv_stopped() delegates to inner SendStream correctly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bidi_stream_recv_stopped_delegates() {
    let mut pair = setup().await;
    let (client_send, mut client_recv, mut server_bidi) =
        client_raw_server_h3(&mut pair).await;

    // Keep client_send alive to prevent premature stream close.
    let _client_send = client_send;

    // Client sends STOP_SENDING on server's send direction (= client recv side).
    client_recv.stop(quinn::VarInt::from_u32(13)).unwrap();

    // Call recv_stopped() directly on the BidiStream (tests delegation).
    match quic::SendStream::<Bytes>::recv_stopped(&mut server_bidi).await {
        Some(StreamErrorIncoming::StreamTerminated { error_code }) => {
            assert_eq!(error_code, 13);
        }
        other => panic!("expected StreamTerminated(13), got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Edge case: recv_reset() returns Pending when stream is borrowed by poll_data
// ---------------------------------------------------------------------------

/// When the inner quinn::RecvStream is borrowed by an in-flight poll_data,
/// recv_reset() should return a pending future.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recv_reset_pending_when_stream_borrowed() {
    let mut pair = setup().await;
    let (_client_send, _client_recv, server_bidi) = client_raw_server_h3(&mut pair).await;
    let (_server_send, mut server_recv) = server_bidi.split();

    // Drain the sentinel byte first.
    let _ = futures_util::future::poll_fn(|cx| quic::RecvStream::poll_data(&mut server_recv, cx))
        .await
        .unwrap();

    // Now start a poll_data that will return Pending (no more data yet).
    {
        let waker = futures_util::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let poll = quic::RecvStream::poll_data(&mut server_recv, &mut cx);
        assert!(poll.is_pending(), "expected poll_data to be Pending");
    }

    // Now self.stream is None (taken by read_chunk_fut).
    // recv_reset() should return a future that is immediately Pending.
    let mut reset_fut = Box::pin(server_recv.recv_reset());
    let waker = futures_util::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    let poll = Pin::new(&mut reset_fut).poll(&mut cx);
    assert!(
        poll.is_pending(),
        "expected recv_reset() to be Pending when stream is borrowed, got {:?}",
        poll
    );
}
