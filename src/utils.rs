pub fn timed<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    tracing::debug!("{} took {:?}", name, elapsed);
    result
}
