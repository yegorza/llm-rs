fn main() {
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");
    
    #[cfg(feature = "napi-binding")]
    napi_build::setup();
}