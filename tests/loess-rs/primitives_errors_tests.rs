#![cfg(feature = "dev")]

use loess_rs::internals::primitives::errors::LoessError;

#[test]
fn test_loess_error_display() {
    // EmptyInput
    let err = LoessError::EmptyInput;
    assert_eq!(format!("{}", err), "Input arrays are empty");

    // InvalidInput
    let err = LoessError::InvalidInput("test error".to_string());
    assert_eq!(format!("{}", err), "Invalid input: test error");

    // MismatchedInputs
    let err = LoessError::MismatchedInputs {
        x_len: 10,
        y_len: 5,
    };
    assert_eq!(
        format!("{}", err),
        "Length mismatch: x has 10 points, y has 5"
    );

    // InvalidNumericalValue
    let err = LoessError::InvalidNumericValue("NaN detected".to_string());
    assert_eq!(format!("{}", err), "Invalid numeric value: NaN detected");

    // TooFewPoints
    let err = LoessError::TooFewPoints { got: 3, min: 5 };
    assert_eq!(format!("{}", err), "Too few points: got 3, need at least 5");

    // InvalidFraction
    let err = LoessError::InvalidFraction(1.5);
    assert_eq!(
        format!("{}", err),
        "Invalid fraction: 1.5 (must be > 0 and <= 1)"
    );

    // InvalidIterations
    let err = LoessError::InvalidIterations(1001);
    assert_eq!(
        format!("{}", err),
        "Invalid iterations: 1001 (must be in [0, 1000])"
    );

    // InvalidIntervals
    let err = LoessError::InvalidIntervals(1.5);
    assert_eq!(
        format!("{}", err),
        "Invalid interval level: 1.5 (must be > 0 and < 1)"
    );

    // InvalidTolerance
    let err = LoessError::InvalidTolerance(-1.0);
    assert_eq!(
        format!("{}", err),
        "Invalid tolerance: -1 (must be > 0 and finite)"
    );

    // InvalidChunkSize
    let err = LoessError::InvalidChunkSize { got: 5, min: 10 };
    assert_eq!(
        format!("{}", err),
        "Invalid chunk_size: 5 (must be at least 10)"
    );

    // InvalidOverlap
    let err = LoessError::InvalidOverlap {
        overlap: 10,
        chunk_size: 10,
    };
    assert_eq!(
        format!("{}", err),
        "Invalid overlap: 10 (must be less than chunk_size 10)"
    );

    // InvalidWindowCapacity
    let err = LoessError::InvalidWindowCapacity { got: 5, min: 10 };
    assert_eq!(
        format!("{}", err),
        "Invalid window_capacity: 5 (must be at least 10)"
    );

    // InvalidMinPoints
    let err = LoessError::InvalidMinPoints {
        got: 1,
        window_capacity: 10,
    };
    assert_eq!(
        format!("{}", err),
        "Invalid min_points: 1 (must be between 2 and window_capacity 10)"
    );

    // UnsupportedFeature
    let err = LoessError::UnsupportedFeature {
        adapter: "Online",
        feature: "CV",
    };
    assert_eq!(
        format!("{}", err),
        "Adapter 'Online' does not support feature: CV"
    );

    // DuplicateParameter
    let err = LoessError::DuplicateParameter { parameter: "foo" };
    assert_eq!(
        format!("{}", err),
        "Parameter 'foo' was set multiple times. Each parameter can only be configured once."
    );

    // InvalidCell
    let err = LoessError::InvalidCell(0.0);
    assert_eq!(
        format!("{}", err),
        "Invalid cell size: 0 (must be in range (0, 1])"
    );

    // InsufficientVertices
    let err = LoessError::InsufficientVertices {
        required: 100,
        limit: 50,
        cell: 0.1,
        cell_provided: true,
        limit_provided: false,
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("Insufficient vertices"),
        "Message was: {}",
        msg
    );
    assert!(msg.contains("100"), "Message was: {}", msg);

    // InsufficientVertices (alternate path)
    let err = LoessError::InsufficientVertices {
        required: 100,
        limit: 50,
        cell: 0.1,
        cell_provided: false,
        limit_provided: true, // trigger !cell_provided && limit_provided path
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains("does not work with user-provided limit"),
        "Message was: {}",
        msg
    );
}

#[test]
fn test_loess_error_properties() {
    let err1 = LoessError::EmptyInput;
    let err2 = err1.clone();
    assert_eq!(err1, err2);
    assert_ne!(err1, LoessError::InvalidInput("foo".to_string()));
}

#[cfg(feature = "std")]
#[test]
fn test_loess_error_is_std_error() {
    fn assert_error<T: std::error::Error>() {}
    assert_error::<LoessError>();
}
