if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
    set(FASTLOESS_ARCH x64)
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
    set(FASTLOESS_ARCH arm64)
else()
    message(
        FATAL_ERROR
        "fastloess does not support ${VCPKG_TARGET_ARCHITECTURE}."
    )
endif()
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

if(VCPKG_TARGET_IS_WINDOWS)
    set(VCPKG_POLICY_ONLY_RELEASE_CRT enabled)
    set(FASTLOESS_BINARY_NAME "fastloess-win32-${FASTLOESS_ARCH}.dll")
    if(FASTLOESS_ARCH STREQUAL "x64")
        set(FASTLOESS_BINARY_SHA512
            a3cc404a77a61066d4301ee38de0c5b7cda04718eb583f2025e91655363098fc79607632fbeaa44a040d3c440aac9dcc8750a5bd7cbb06be52059249ece0c851
        )
    else()
        set(FASTLOESS_BINARY_SHA512
            16f787b909b3a8f7220de6281a008ea6dd7764c9adcd74ace7cc7ea473cb5106c1c9f53588538ff50ec6eaef7df72957c6495668eff5dbae66b1558aaece123b
        )
    endif()
elseif(VCPKG_TARGET_IS_LINUX)
    if(VCPKG_TARGET_TRIPLET MATCHES "musl")
        message(
            FATAL_ERROR
            "fastloess v${VERSION} has no published musl binary. Use a glibc dynamic triplet."
        )
    endif()
    set(FASTLOESS_BINARY_NAME "libfastloess-linux-${FASTLOESS_ARCH}.so")
    if(FASTLOESS_ARCH STREQUAL "x64")
        set(FASTLOESS_BINARY_SHA512
            71826281d448c259137348440502a3d422dc8b4a7bc25e107751ece6dbbd05902d5bef07f89758f94c870621aa3e668b3f3bd946829af267e5984ec65ce8f366
        )
    else()
        set(FASTLOESS_BINARY_SHA512
            8042f412c7a0b70a24f06e34ee692200b556e35977146e58310e8a563dda1d35ac101d4482216815b4dda79a458f649d59f8f0fd217d0d27cbe93ea42809947f
        )
    endif()
elseif(VCPKG_TARGET_IS_OSX)
    set(FASTLOESS_BINARY_NAME "libfastloess-macos-${FASTLOESS_ARCH}.dylib")
    if(FASTLOESS_ARCH STREQUAL "x64")
        set(FASTLOESS_BINARY_SHA512
            421f6f078ac756c9ed845997eafe3eed57ed4963aa35ffe96211a286e26e85013292b0650f82f3f9bd57864bc62ec1ca97b378eb5e66d54c46b3667d65a49577
        )
    else()
        set(FASTLOESS_BINARY_SHA512
            37746abf1a1f1f335e3548d0c5a22c2acc6aff8cb1fccc66ac6f025627b2010f55b1376c130517876b484a91c4a1034164d09a2315aa6e0a83552dddbf8e8950
        )
    endif()
else()
    message(FATAL_ERROR "fastloess does not support this target platform.")
endif()

vcpkg_download_distfile(
    FASTLOESS_BINARY
    URLS "https://github.com/thisisamirv/loess-project/releases/download/v${VERSION}/${FASTLOESS_BINARY_NAME}"
    FILENAME "fastloess-v${VERSION}/${FASTLOESS_BINARY_NAME}"
    SHA512 "${FASTLOESS_BINARY_SHA512}"
)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO thisisamirv/loess-project
    REF "v${VERSION}"
    SHA512 d317e4dd23c581aee407d7eb99543c37dea99b017702f8a291a470bb6639fdb4c8b64464b0851e6f83cca3699c9fd92edcca222f8f635cfa2edae1881671d0ee
)
set(FASTLOESS_CMAKE_OPTIONS
    "-DFASTLOESS_SOURCE_DIR=${SOURCE_PATH}"
    "-DFASTLOESS_BINARY=${FASTLOESS_BINARY}"
)
if(VCPKG_TARGET_IS_WINDOWS)
    list(APPEND FASTLOESS_CMAKE_OPTIONS "-DFASTLOESS_ARCH=${FASTLOESS_ARCH}")
endif()
vcpkg_cmake_configure(
    SOURCE_PATH "${CURRENT_PORT_DIR}"
    OPTIONS ${FASTLOESS_CMAKE_OPTIONS}
)
vcpkg_cmake_install()
file(
    REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)
vcpkg_install_copyright(
    FILE_LIST "${SOURCE_PATH}/LICENSE-MIT" "${SOURCE_PATH}/LICENSE-APACHE"
    COMMENT "The prebuilt native library statically links Rust dependencies. The v${VERSION} release does not publish its original binary-build Cargo.lock or dependency license report; running cargo-about later cannot reconstruct the dependency graph embedded in these binaries. Exact dependency-version provenance is therefore unavailable. Future releases build with the committed workspace lockfile and include THIRD_PARTY_LICENSES.html."
)
file(
    INSTALL "${CURRENT_PORT_DIR}/usage"
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)
