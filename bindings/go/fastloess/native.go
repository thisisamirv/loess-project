//go:build !external_native

package fastloess

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo linux,amd64,!musl LDFLAGS: -L${SRCDIR}/native/linux_amd64 -lfastloess_go -lm -ldl -lpthread
#cgo linux,arm64,!musl LDFLAGS: -L${SRCDIR}/native/linux_arm64 -lfastloess_go -lm -ldl -lpthread
#cgo linux,amd64,musl LDFLAGS: -L${SRCDIR}/native/linux_amd64_musl -lfastloess_go -lm -ldl -lpthread
#cgo linux,arm64,musl LDFLAGS: -L${SRCDIR}/native/linux_arm64_musl -lfastloess_go -lm -ldl -lpthread
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/native/darwin_amd64 -lfastloess_go
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/native/darwin_arm64 -lfastloess_go
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/native/windows_amd64 -lfastloess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
#cgo windows,arm64 LDFLAGS: -static -L${SRCDIR}/native/windows_arm64 -lfastloess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
*/
import "C"
