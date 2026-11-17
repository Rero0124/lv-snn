// 빌드 식별자를 빌드 시점에 넣는다: 커밋 해시, 커밋 안 된 변경이 있는지, 빌드 시각.
// snn 과 fly 가 같이 쓴다 (fly/Cargo.toml 의 build = "../snn/build.rs"). 실행 기록 첫 줄에 찍힌다.
use std::process::Command;

fn run(dir: &str, cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).current_dir(dir).output().ok()?;
    if !out.status.success() { return None; }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    // 저장소 뿌리 = 크레이트 폴더의 위
    let root = format!("{}/..", std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // 두 실행 파일을 만드는 데 들어가는 것들 (fly 는 snn/src 의 일부를 같이 쓴다)
    let sources = ["snn/src", "snn/build.rs", "snn/Cargo.toml", "snn/Cargo.lock", "fly/src", "fly/Cargo.toml", "fly/Cargo.lock"];
    let hash = run(&root, "git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "커밋 모름".into());
    // --no-optional-locks: 상태를 보는 것만으로 .git/index 가 바뀌어 다시 빌드되는 일을 막는다
    let mut status = vec!["--no-optional-locks", "status", "--porcelain", "--"];
    status.extend(sources);
    let dirty = match run(&root, "git", &status) {
        Some(s) if s.is_empty() => "",
        Some(_) => " + 커밋 안 된 변경",
        None => " (변경 여부 모름)",
    };
    let when = run(&root, "date", &["+%Y-%m-%d %H:%M:%S"]).unwrap_or_default();
    println!("cargo:rustc-env=BUILD_ID={hash}{dirty}, 빌드 {when}");
    // 커밋이 바뀌거나 소스가 바뀌면 다시 계산한다
    for p in [".git/HEAD", ".git/index"].iter().chain(sources.iter()) {
        println!("cargo:rerun-if-changed={root}/{p}");
    }
}
