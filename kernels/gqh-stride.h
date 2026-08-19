#pragma once
// 4-plane GQH device layout. On-disk / CPU payload stays the tight
// 105 / 73 / 66-byte superblock. SuperSonic planarizes at load.
//
// Per row, indexed by superblock sb:
//   d[sb]          1 B
//   ratio[sb << 3] 8 B
//   lo[sb << 6]    64 B
//   hi[sb << 5]    32 B   (GQH3 only)
// FLM can emit this natively later.

#include "gqh-tables.h"

#define GQH_PLANE_ALIGN 64
// Decode-only padded AoS: 7 bytes after ratio so lo starts at 16.
// Product bytes are unchanged; pad is unused. 16-byte SB alignment lets
// the GEMV issue aligned ushort/hi loads instead of odd-offset memcpy.
#define GQH2H_PAD_SB_BYTES 80
#define GQH3_PAD_SB_BYTES  112
#define GQH_PAD_LO           16
#define GQH_PAD_HI           80

#ifdef __HIPCC__
#define GQH_HD __host__ __device__
#else
#define GQH_HD
#endif

GQH_HD inline void gqh_plane_offsets(
    int nsb, int is3, int* off_ratio, int* off_lo, int* off_hi, int* stride) {
    int o = (nsb + 7) & ~7;
    *off_ratio = o;
    o += nsb << 3;
    o = (o + (GQH_PLANE_ALIGN - 1)) & ~(GQH_PLANE_ALIGN - 1);
    *off_lo = o;
    o += nsb << 6;
    *off_hi = o;
    if (is3) {
        o += nsb << 5;
    }
    *stride = (o + (GQH_PLANE_ALIGN - 1)) & ~(GQH_PLANE_ALIGN - 1);
}

GQH_HD inline int gqh_plane_row_bytes(int nsb, int is3) {
    int off_ratio, off_lo, off_hi, stride;
    gqh_plane_offsets(nsb, is3, &off_ratio, &off_lo, &off_hi, &stride);
    return stride;
}
