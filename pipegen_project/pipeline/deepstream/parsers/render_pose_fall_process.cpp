#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

// ─── Per-person detection ────────────────────────────────────────────────────
struct DetKpt { float x, y, v; };
struct Detection {
    float x1, y1, x2, y2, score;
    DetKpt kpts[17];
};

// ─── Tracker state ───────────────────────────────────────────────────────────
struct TrackBox { float x1, y1, x2, y2; };
struct TrackState {
    TrackBox box;
    int missed;
};

// ─── Fall state per track ────────────────────────────────────────────────────
struct FallState {
    bool has_floor_y;   float floor_y;
    bool has_prev_hip;  float prev_hip_x, prev_hip_y;
    bool has_prev_t;    double prev_t;
    bool has_drop_t;    double drop_t;
    bool has_hold_start;double hold_start;
    bool fallen;
};

// ─── Person result ───────────────────────────────────────────────────────────
struct PersonResult {
    int id;
    float box[4]; // x1 y1 x2 y2
    DetKpt kpts[17];
    float score;
    bool fallen;
    bool fall_event;
    const char *fall_status; // pointer to static string
};

// ─── Persistent state ─── We use function-local statics, but since the pipeline
// has exactly one instance we are fine. Wrapped in a struct for clarity.
struct PersistentState {
    std::map<int, TrackState> tracks;
    int next_id = 1;
    std::map<int, FallState> fall_states;
    bool has_video_start = false;
    double video_start_s = 0.0;
    uint64_t frame_index = 0;
};

static PersistentState &get_state() {
    static PersistentState s;
    return s;
}

// ─── IoU ─────────────────────────────────────────────────────────────────────
static float compute_iou(const float *a, const TrackBox &b) {
    float ix1 = std::max(a[0], b.x1), iy1 = std::max(a[1], b.y1);
    float ix2 = std::min(a[2], b.x2), iy2 = std::min(a[3], b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float aa = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]);
    float ab = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    float uni = aa + ab - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// ─── Pixelate helper on a cv::Mat region ─────────────────────────────────────
static void pixelate_region(cv::Mat &plane, int x0, int y0, int x1, int y1, int cell) {
    if (x0 >= x1 || y0 >= y1) return;
    x0 = std::max(x0, 0); y0 = std::max(y0, 0);
    x1 = std::min(x1, plane.cols); y1 = std::min(y1, plane.rows);
    if (x1 - x0 < 2 || y1 - y0 < 2) return;
    cv::Mat block = plane(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    int bw = block.cols, bh = block.rows;
    int sw = std::max(1, bw / cell), sh = std::max(1, bh / cell);
    cv::Mat small;
    cv::resize(block, small, cv::Size(sw, sh), 0, 0, cv::INTER_AREA);
    cv::resize(small, block, cv::Size(bw, bh), 0, 0, cv::INTER_NEAREST);
}

// ─── COCO skeleton ───────────────────────────────────────────────────────────
static const int SKELETON[][2] = {
    {15,13},{13,11},{16,14},{14,12},{11,12},{5,11},{6,12},{5,6},
    {5,7},{6,8},{7,9},{8,10},{1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6}
};
static const int NSKEL = 19;
static const int NKPT = 17;

static const char* CLASS_NAMES[] = {"person"};

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[render_pose_fall] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) rs;

    PersistentState &S = get_state();

    // Get timestamp from frame_meta
    NvDsFrameMeta *fmeta = (NvDsFrameMeta *) in->frame_meta;
    double timestamp_s = 0.0;
    if (fmeta) {
        // nv_ds_frame_meta buf_pts is in nanoseconds
        timestamp_s = (double)fmeta->buf_pts / 1.0e9;
    }
    uint64_t frame_index = S.frame_index++;

    if (!S.has_video_start) {
        S.has_video_start = true;
        S.video_start_s = timestamp_s;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 1. TENSOR DECODE
    // ═══════════════════════════════════════════════════════════════════════
    std::vector<Detection> detections;

    auto ts = collect_tensors(in);
    // The model output is [1, 8400, 56] — after batch strip: [8400, 56]
    const TensorRef *tref = find_tensor(ts, 8400, 56);
    if (!tref) {
        // Try without batch strip: shape might be [1, 8400, 56] -> ndim=3, d={1,8400,56}
        // or the nvinfer might report [56, 8400] if transposed. Check both.
        tref = find_tensor(ts, 56, 8400);
    }

    // Compute letterbox geometry at runtime
    const float net_w = 640.0f, net_h = 640.0f;
    const float gain = std::min(net_w / (float)W, net_h / (float)H);
    const float pad_w = (net_w - (float)W * gain) * 0.5f;
    const float pad_h = (net_h - (float)H * gain) * 0.5f;

    if (tref) {
        const float *data = tref->data;
        bool transposed = false;
        int nAnchors = 8400;
        int nCols = 56;

        if (tref->d[0] == 56 && tref->d[1] == 8400) {
            transposed = true;
            nAnchors = 8400;
            nCols = 56;
        } else if (tref->d[0] == 8400 && tref->d[1] == 56) {
            transposed = false;
            nAnchors = 8400;
            nCols = 56;
        }

        // Collect candidates above confidence threshold
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_indices; // index into detections_raw

        struct RawDet {
            float x1, y1, x2, y2, conf;
            float kx[17], ky[17], kv[17];
        };
        std::vector<RawDet> detections_raw;

        for (int i = 0; i < nAnchors; ++i) {
            float bx1, by1, bx2, by2, conf;
            if (!transposed) {
                const float *row = data + (size_t)i * nCols;
                bx1 = row[0]; by1 = row[1]; bx2 = row[2]; by2 = row[3];
                conf = row[4];
            } else {
                bx1 = data[0 * nAnchors + i];
                by1 = data[1 * nAnchors + i];
                bx2 = data[2 * nAnchors + i];
                by2 = data[3 * nAnchors + i];
                conf = data[4 * nAnchors + i];
            }
            if (conf < 0.25f) continue;

            RawDet rd;
            rd.x1 = bx1; rd.y1 = by1; rd.x2 = bx2; rd.y2 = by2; rd.conf = conf;
            for (int k = 0; k < 17; ++k) {
                if (!transposed) {
                    const float *row = data + (size_t)i * nCols;
                    rd.kx[k] = row[5 + k * 3 + 0];
                    rd.ky[k] = row[5 + k * 3 + 1];
                    rd.kv[k] = row[5 + k * 3 + 2];
                } else {
                    rd.kx[k] = data[(5 + k * 3 + 0) * nAnchors + i];
                    rd.ky[k] = data[(5 + k * 3 + 1) * nAnchors + i];
                    rd.kv[k] = data[(5 + k * 3 + 2) * nAnchors + i];
                }
            }
            int idx = (int)detections_raw.size();
            detections_raw.push_back(rd);

            // NMS uses xywh format
            float bw = bx2 - bx1, bh = by2 - by1;
            nms_boxes.push_back(cv::Rect((int)bx1, (int)by1, (int)bw, (int)bh));
            nms_scores.push_back(conf);
            nms_indices.push_back(idx);
        }

        // Single-class NMS
        std::vector<int> keep;
        if (!nms_boxes.empty()) {
            cv::dnn::NMSBoxes(nms_boxes, nms_scores, 0.25f, 0.45f, keep);
        }

        for (int ki = 0; ki < (int)keep.size(); ++ki) {
            int ri = nms_indices[keep[ki]];
            const RawDet &rd = detections_raw[ri];
            Detection det;
            // Transform from 640-space to frame-space
            det.x1 = std::max(0.0f, (rd.x1 - pad_w) / gain);
            det.y1 = std::max(0.0f, (rd.y1 - pad_h) / gain);
            det.x2 = std::min((float)W, (rd.x2 - pad_w) / gain);
            det.y2 = std::min((float)H, (rd.y2 - pad_h) / gain);
            det.score = rd.conf;
            for (int k = 0; k < 17; ++k) {
                det.kpts[k].x = (rd.kx[k] - pad_w) / gain;
                det.kpts[k].y = (rd.ky[k] - pad_h) / gain;
                det.kpts[k].v = rd.kv[k];
            }
            detections.push_back(det);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 2. IoU TRACKING
    // ═══════════════════════════════════════════════════════════════════════
    struct Candidate { float iou_val; int tid; int det_idx; };
    std::vector<Candidate> candidates;
    for (auto &tp : S.tracks) {
        for (int j = 0; j < (int)detections.size(); ++j) {
            float b[4] = {detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2};
            float iov = compute_iou(b, tp.second.box);
            if (iov >= 0.3f) {
                candidates.push_back({iov, tp.first, j});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        return a.iou_val > b.iou_val;
    });

    std::map<int, int> assignment; // det_idx -> track_id
    std::map<int, bool> used_ids;
    std::map<int, bool> used_dets;
    for (auto &c : candidates) {
        if (used_ids.count(c.tid) || used_dets.count(c.det_idx)) continue;
        assignment[c.det_idx] = c.tid;
        used_ids[c.tid] = true;
        used_dets[c.det_idx] = true;
    }

    // Update missed tracks
    for (auto it = S.tracks.begin(); it != S.tracks.end(); ) {
        if (!used_ids.count(it->first)) {
            it->second.missed++;
            if (it->second.missed > 30) {
                S.fall_states.erase(it->first);
                it = S.tracks.erase(it);
                continue;
            }
        }
        ++it;
    }

    // Build current results with assigned IDs
    struct TrackedDet {
        int id;
        Detection det;
    };
    std::vector<TrackedDet> current;
    for (int j = 0; j < (int)detections.size(); ++j) {
        int tid;
        if (assignment.count(j)) {
            tid = assignment[j];
        } else {
            tid = S.next_id++;
        }
        TrackState ts_new;
        ts_new.box = {detections[j].x1, detections[j].y1, detections[j].x2, detections[j].y2};
        ts_new.missed = 0;
        S.tracks[tid] = ts_new;
        current.push_back({tid, detections[j]});
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 3. FALL DETECTION
    // ═══════════════════════════════════════════════════════════════════════
    // Reset state for missed tracks (those in fall_states but not in current)
    {
        std::map<int, bool> present_ids;
        for (auto &c : current) present_ids[c.id] = true;
        for (auto &fs : S.fall_states) {
            if (!present_ids.count(fs.first)) {
                fs.second.has_hold_start = false;
                fs.second.has_prev_hip = false;
                fs.second.has_prev_t = false;
                fs.second.has_drop_t = false;
                fs.second.fallen = false;
            }
        }
    }

    std::vector<PersonResult> results;
    double t = timestamp_s;

    for (auto &c : current) {
        int tid = c.id;
        Detection &det = c.det;

        if (!S.fall_states.count(tid)) {
            FallState fs;
            fs.has_floor_y = false; fs.floor_y = 0.0f;
            fs.has_prev_hip = false; fs.prev_hip_x = 0.0f; fs.prev_hip_y = 0.0f;
            fs.has_prev_t = false; fs.prev_t = 0.0;
            fs.has_drop_t = false; fs.drop_t = 0.0;
            fs.has_hold_start = false; fs.hold_start = 0.0;
            fs.fallen = false;
            S.fall_states[tid] = fs;
        }
        FallState &st = S.fall_states[tid];

        // Check pose validity: both shoulders (5,6) and both hips (11,12)
        bool pose_valid = (det.kpts[5].v >= 0.5f && det.kpts[6].v >= 0.5f &&
                           det.kpts[11].v >= 0.5f && det.kpts[12].v >= 0.5f);

        float hip_x = 0, hip_y = 0;
        float torso_len = 0;
        float angle_deg = -1.0f; // negative = invalid
        bool has_hip = false;

        if (pose_valid) {
            float sh_x = (det.kpts[5].x + det.kpts[6].x) * 0.5f;
            float sh_y = (det.kpts[5].y + det.kpts[6].y) * 0.5f;
            hip_x = (det.kpts[11].x + det.kpts[12].x) * 0.5f;
            hip_y = (det.kpts[11].y + det.kpts[12].y) * 0.5f;
            has_hip = true;
            float dx = hip_x - sh_x;
            float dy = hip_y - sh_y;
            torso_len = std::sqrt(dx * dx + dy * dy);
            if (torso_len > 0.0f) {
                angle_deg = (float)(std::atan2(std::fabs(dx), std::fabs(dy) + 1e-6) * 180.0 / M_PI);
            }
        }

        // Floor calibration: first 3 seconds, upright (angle <= 30)
        if (t - S.video_start_s <= 3.0 && angle_deg >= 0.0f && angle_deg <= 30.0f) {
            for (int ai = 0; ai < 2; ++ai) {
                int idx = (ai == 0) ? 15 : 16;
                if (det.kpts[idx].v >= 0.5f) {
                    float ay = det.kpts[idx].y;
                    if (!st.has_floor_y || ay > st.floor_y) {
                        st.floor_y = ay;
                        st.has_floor_y = true;
                    }
                }
            }
        }

        // Rapid hip drop
        bool rapid_drop = false;
        if (has_hip && torso_len > 0.0f && st.has_prev_hip && st.has_prev_t) {
            double dt = t - st.prev_t;
            if (dt > 0.0) {
                float speed = (float)((hip_y - st.prev_hip_y) / (torso_len * dt));
                rapid_drop = speed > 1.2f;
            }
        }
        if (rapid_drop) {
            st.has_drop_t = true;
            st.drop_t = t;
        }

        // Expire sequence after 1.5s
        if (st.has_drop_t && t - st.drop_t > 1.5) {
            st.has_drop_t = false;
            st.has_hold_start = false;
        }

        // Near floor check
        bool near_floor = false;
        if (has_hip && torso_len > 0.0f && st.has_floor_y) {
            near_floor = ((st.floor_y - hip_y) / torso_len) < 0.5f;
        }

        bool posture = (angle_deg > 60.0f && near_floor);

        // Posture + drop_t -> hold_start
        if (st.has_drop_t && posture && (t - st.drop_t <= 1.5) && !st.has_hold_start) {
            st.has_hold_start = true;
            st.hold_start = t;
        } else if (st.has_hold_start && !posture) {
            st.has_hold_start = false;
            st.has_drop_t = false;
            st.fallen = false;
        }

        bool fall_event = false;
        if (st.has_hold_start && posture && (t - st.hold_start >= 3.0) && !st.fallen) {
            st.fallen = true;
            fall_event = true;
        }

        // Update prev_hip / prev_t
        if (has_hip && torso_len > 0.0f) {
            st.has_prev_hip = true;
            st.prev_hip_x = hip_x;
            st.prev_hip_y = hip_y;
            st.has_prev_t = true;
            st.prev_t = t;
        } else {
            st.has_prev_hip = false;
            st.has_prev_t = false;
        }

        const char *status;
        if (!st.has_floor_y) status = "floor_uncalibrated";
        else if (st.fallen) status = "fallen";
        else if (st.has_hold_start) status = "fall_posture_pending";
        else status = "observing";

        PersonResult pr;
        pr.id = tid;
        pr.box[0] = det.x1; pr.box[1] = det.y1; pr.box[2] = det.x2; pr.box[3] = det.y2;
        memcpy(pr.kpts, det.kpts, sizeof(det.kpts));
        pr.score = det.score;
        pr.fallen = st.fallen;
        pr.fall_event = fall_event;
        pr.fall_status = status;
        results.push_back(pr);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 4. PEOPLE BLUR (pixelation on BGR frame)
    // ═══════════════════════════════════════════════════════════════════════
    cv::Mat frame = read_nv12_bgr(in);

    // Convert to I420 for separate Y/UV pixelation, as per the reference
    {
        cv::Mat i420;
        cv::cvtColor(frame, i420, cv::COLOR_BGR2YUV_I420);
        int n = W * H;
        // Y plane: H rows x W cols
        cv::Mat yplane(H, W, CV_8UC1, i420.data);
        // U and V planes
        uint8_t *udata = i420.data + n;
        uint8_t *vdata = udata + n / 4;
        int cW = W / 2, cH = H / 2;
        cv::Mat uplane(cH, cW, CV_8UC1, udata);
        cv::Mat vplane(cH, cW, CV_8UC1, vdata);
        // Interleave U/V into 2-channel for pixelation
        cv::Mat uvplane(cH, cW, CV_8UC2);
        for (int r = 0; r < cH; ++r) {
            for (int c = 0; c < cW; ++c) {
                uvplane.at<cv::Vec2b>(r, c)[0] = uplane.at<uint8_t>(r, c);
                uvplane.at<cv::Vec2b>(r, c)[1] = vplane.at<uint8_t>(r, c);
            }
        }

        const int cell = 16;
        for (auto &pr : results) {
            // Even-aligned box
            int bx0 = std::max(0, ((int)std::floor(pr.box[0] / 2.0f)) * 2);
            int by0 = std::max(0, ((int)std::floor(pr.box[1] / 2.0f)) * 2);
            int bx1 = std::min(W, ((int)std::ceil(pr.box[2] / 2.0f)) * 2);
            int by1 = std::min(H, ((int)std::ceil(pr.box[3] / 2.0f)) * 2);
            if (bx1 - bx0 < 2 || by1 - by0 < 2) continue;

            // Pixelate Y
            pixelate_region(yplane, bx0, by0, bx1, by1, cell);
            // Pixelate UV (half res)
            pixelate_region(uvplane, bx0 / 2, by0 / 2, bx1 / 2, by1 / 2, std::max(1, cell / 2));
        }

        // Reconstruct NV12 -> BGR
        // Build NV12 from Y + interleaved UV
        cv::Mat nv12(H + cH, W, CV_8UC1);
        for (int r = 0; r < H; ++r)
            memcpy(nv12.ptr(r), yplane.ptr(r), (size_t)W);
        for (int r = 0; r < cH; ++r) {
            uint8_t *dst = nv12.ptr(H + r);
            for (int c = 0; c < cW; ++c) {
                dst[2 * c] = uvplane.at<cv::Vec2b>(r, c)[0];
                dst[2 * c + 1] = uvplane.at<cv::Vec2b>(r, c)[1];
            }
        }
        cv::cvtColor(nv12, frame, cv::COLOR_YUV2BGR_NV12);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 5. SKELETON DRAWING (on the pixelated BGR frame)
    // ═══════════════════════════════════════════════════════════════════════
    for (auto &pr : results) {
        cv::Scalar boxColor = pr.fallen ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        int ix1 = (int)pr.box[0], iy1 = (int)pr.box[1];
        int ix2 = (int)pr.box[2], iy2 = (int)pr.box[3];
        cv::rectangle(frame, cv::Point(ix1, iy1), cv::Point(ix2, iy2), boxColor, 2);

        // Label: "person <conf>%" and "ID <id> <status>"
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s %d%%", CLASS_NAMES[0], (int)(pr.score * 100.0f + 0.5f));
            cv::putText(frame, buf, cv::Point(ix1, std::max(0, iy1 - 24)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, boxColor, 2, cv::LINE_AA);
        }
        {
            char buf2[128];
            if (pr.fallen)
                snprintf(buf2, sizeof(buf2), "ID %d FALL", pr.id);
            else
                snprintf(buf2, sizeof(buf2), "ID %d %s", pr.id, pr.fall_status);
            cv::putText(frame, buf2, cv::Point(ix1, std::max(0, iy1 - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, boxColor, 2, cv::LINE_AA);
        }

        // Skeleton limbs
        for (int s = 0; s < NSKEL; ++s) {
            int a = SKELETON[s][0], b = SKELETON[s][1];
            if (pr.kpts[a].v >= 0.5f && pr.kpts[b].v >= 0.5f) {
                cv::line(frame,
                         cv::Point((int)pr.kpts[a].x, (int)pr.kpts[a].y),
                         cv::Point((int)pr.kpts[b].x, (int)pr.kpts[b].y),
                         cv::Scalar(255, 200, 0), 2, cv::LINE_AA);
            }
        }
        // Keypoint dots
        for (int k = 0; k < NKPT; ++k) {
            if (pr.kpts[k].v >= 0.5f) {
                cv::circle(frame, cv::Point((int)pr.kpts[k].x, (int)pr.kpts[k].y),
                           3, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
            }
        }
    }

    // Write back to NV12
    write_nv12(frame, in);

    // ═══════════════════════════════════════════════════════════════════════
    // 6. METADATA PUBLISHING
    // ═══════════════════════════════════════════════════════════════════════
    render_meta::Builder sb(W, H);

    for (auto &pr : results) {
        // Skeleton
        render_meta::Skeleton sk(NKPT);
        for (int k = 0; k < NKPT; ++k) {
            if (pr.kpts[k].v >= 0.5f)
                sk.set((size_t)k, (double)pr.kpts[k].x, (double)pr.kpts[k].y);
        }
        for (int s = 0; s < NSKEL; ++s) {
            sk.link((size_t)SKELETON[s][0], (size_t)SKELETON[s][1]);
        }
        sb.skeleton(sk, CLASS_NAMES[0], pr.id, (double)pr.score);

        // Box
        sb.box((double)pr.box[0], (double)pr.box[1],
               (double)(pr.box[2] - pr.box[0]), (double)(pr.box[3] - pr.box[1]),
               CLASS_NAMES[0], pr.id, (double)pr.score);

        // Fall fields per person (use field_s / field_i)
        // We encode per-person fields with a tag
        {
            char key_fallen[64], key_status[64], key_event[64];
            snprintf(key_fallen, sizeof(key_fallen), "person_%d_fallen", pr.id);
            snprintf(key_status, sizeof(key_status), "person_%d_fall_status", pr.id);
            snprintf(key_event, sizeof(key_event), "person_%d_fall_event", pr.id);
            sb.field_i(key_fallen, pr.fallen ? 1 : 0);
            sb.field_s(key_status, pr.fall_status);
            sb.field_i(key_event, pr.fall_event ? 1 : 0);
        }
    }

    sb.field_i("frame_index", (int64_t)frame_index);
    sb.field_d("timestamp_s", timestamp_s);
    sb.field_i("num_persons", (int64_t)results.size());

    sb.publish(in->frame_meta);

    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);