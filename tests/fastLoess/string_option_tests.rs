#![cfg(feature = "dev")]
use fastLoess::prelude::*;

// ─── String arguments accepted by builder ────────────────────────────────────

#[test]
fn test_string_boundary_policy_lowercase() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .boundary_policy("reflect")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_boundary_policy_mixed_case() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .boundary_policy("Reflect")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_boundary_policy_uppercase() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .boundary_policy("REFLECT")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_alias_pad_equals_extend() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    // "pad" is an alias for "extend"
    let res = Loess::new()
        .boundary_policy("pad")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_weight_function() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .weight_function("epanechnikov")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_alias_boxcar_equals_uniform() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    // "boxcar" is an alias for "uniform"
    let res = Loess::new()
        .weight_function("boxcar")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_robustness_method() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .robustness_method("huber")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_scaling_method() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .scaling_method("mar")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_degree() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .degree("quadratic")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_degree_numeric() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    // "2" is equivalent to "quadratic"
    let res = Loess::new()
        .degree("2")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_surface_mode() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .surface_mode("direct")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_zero_weight_fallback() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .zero_weight_fallback("return_original")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

#[test]
fn test_string_merge_strategy() {
    let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| xi * 2.0).collect();
    let res = StreamingLoess::new()
        .merge_strategy("average")
        .chunk_size(20)
        .overlap(0)
        .build()
        .unwrap()
        .process_chunk(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 20);
}

#[test]
fn test_string_update_mode() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let mut model = OnlineLoess::new().update_mode("full").build().unwrap();
    for (xi, yi) in x.iter().zip(y.iter()) {
        model.add_point(&[*xi], *yi).unwrap();
    }
}

// ─── Enum variants still work unchanged ──────────────────────────────────────

#[test]
fn test_enum_variant_still_works() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let res = Loess::new()
        .boundary_policy("reflect")
        .weight_function("epanechnikov")
        .robustness_method("huber")
        .scaling_method("mar")
        .degree("quadratic")
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}

// ─── Invalid strings → build() returns Err(ParseErrors) ─────────────────────

#[test]
fn test_invalid_string_single_error() {
    let result = Loess::new().boundary_policy("totally_invalid").build();
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        matches!(err, LoessError::ParseErrors(_)),
        "expected ParseErrors, got: {err}"
    );
}

#[test]
fn test_invalid_strings_multiple_errors_all_reported() {
    let result = Loess::new()
        .boundary_policy("bad_policy")
        .weight_function("bad_kernel")
        .robustness_method("bad_method")
        .build();
    assert!(result.is_err());
    match result.err().unwrap() {
        LoessError::ParseErrors(errors) => {
            assert_eq!(
                errors.len(),
                3,
                "expected 3 parse errors (one per bad string), got {}: {errors:?}",
                errors.len()
            );
        }
        other => panic!("expected ParseErrors, got: {other}"),
    }
}

// ─── ParseErrors Display contains all messages ───────────────────────────────

#[test]
fn test_parse_errors_display_contains_all_messages() {
    let result = Loess::new()
        .boundary_policy("bad_policy")
        .degree("bad_degree")
        .build();
    let msg = result.err().unwrap().to_string();
    // Display should mention both errors
    assert!(msg.contains("[0]"), "Display missing error 0: {msg}");
    assert!(msg.contains("[1]"), "Display missing error 1: {msg}");
}

// ─── String type (String, not &str) also accepted ────────────────────────────

#[test]
fn test_owned_string_accepted() {
    let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0_f64, 4.0, 6.0, 8.0, 10.0];
    let policy = String::from("reflect");
    let res = Loess::new()
        .boundary_policy(&policy)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    assert_eq!(res.y.len(), 5);
}
