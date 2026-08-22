use clap::error::ErrorKind;
use runner::cli::parse_cli_from;

fn minimal_args() -> [&'static str; 9] {
    [
        "supersonic",
        "--model",
        "qwen3.8-27b",
        "--model-dir",
        "/models/qwen38",
        "--gguf-file",
        "/models/qwen38.gqh.gguf",
        "--prompt",
        "Hello",
    ]
}

#[test]
fn parses_retained_qwen38_command_with_greedy_defaults() {
    let cli = parse_cli_from(minimal_args()).unwrap();

    assert_eq!(cli.model, "qwen3.8-27b");
    assert_eq!(cli.max_new_tokens, 8);
}

#[test]
fn requires_model_dir() {
    let args = [
        "supersonic",
        "--model",
        "qwen3.8-27b",
        "--gguf-file",
        "/models/qwen38.gqh.gguf",
        "--prompt",
        "Hello",
    ];

    let error = parse_cli_from(args).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
}

#[test]
fn requires_gguf_file() {
    let args = [
        "supersonic",
        "--model",
        "qwen3.8-27b",
        "--model-dir",
        "/models/qwen38",
        "--prompt",
        "Hello",
    ];

    let error = parse_cli_from(args).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
}

#[test]
fn rejects_unsupported_model_name() {
    let mut args = minimal_args();
    args[2] = "qwen3.5-0.8b";

    let error = parse_cli_from(args).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidValue);
}

#[test]
fn rejects_removed_options_as_unknown_arguments() {
    for option in [
        "--backend",
        "--flm-file",
        "--q4km",
        "--dflash",
        "--specprefill",
        "--certified-kv",
    ] {
        let mut args = minimal_args().to_vec();
        args.push(option);

        let error = parse_cli_from(args).unwrap_err();

        assert_eq!(
            error.kind(),
            ErrorKind::UnknownArgument,
            "{option} should be rejected as an unknown argument"
        );
    }
}
