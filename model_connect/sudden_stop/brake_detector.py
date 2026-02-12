import numpy as np
from . import config as cfg
import itertools


class BrakeDetector:
    
    def __init__(self, fps, img_w=None, img_h=None):
        self.fps = float(fps)
        self.dt = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        self.frame_idx = 0

        self.tracks = {}
        self.events = {}
        self.event_id_gen = itertools.count(1)

        self.frame_history = {}
        self.history_limit = int(self.fps * 30)  # 30초 유지    

        self.LATERAL_SKIP_TH = 1.25
        self.LONG_RANGE_TH = 60.0
        self.WIDTH_ALPHA = 0.4
        self.CUTIN_WIDTH = 2.0

        # Logistic Regression weights
        self.W_iTTC = 1.9194
        self.W_ACC = -0.0105
        self.W_SPEED = -0.0176
        self.BIAS = -1.3964

        self.ALPHA_SPEED = 0.3
        self.ALPHA_ACC = 0.2
        self.MAX_RAW_SPEED = 100.0
        self.MAX_RAW_ACC = 100.0

        self.WARNING_DURATION_TH = 3
        self.EVENT_MERGE_GAP = int(self.fps * 1.5)

    def _init_track(self, tid, dist, pos_x, width_px, aspect_ratio):
        self.tracks[tid] = {
            "dist": float(dist),
            "pos_x": float(pos_x),
            "prev_pos_x": float(pos_x),
            "speed": 0.0,
            "acc": 0.0,
            "raw_acc": 0.0,
            "ttc": 99.9,
            "width": float(width_px),
            "aspect_ratio": float(aspect_ratio),
            "cnt": 1,
            "warning_cnt": 0,
            "warning_type": cfg.WARN_NORMAL,
            "lateral_speed": 0.0,
            "last_event_id": None
        }

    def _calculate_ai_score(self, ttc, acc, speed_mps):
        if ttc <= 0.001:
            i_ttc = 100.0
        else:
            clamped_ttc = min(ttc, 10.0)
            i_ttc = 1.0 / (clamped_ttc + 0.1)

        speed_kmh = speed_mps * 3.6
        return (
            self.W_iTTC * i_ttc
            + self.W_ACC * acc
            + self.W_SPEED * speed_kmh
            + self.BIAS
        )

    def _get_or_create_event(self, tid, dist, pos_x):
        for eid, ev in self.events.items():
            if ev["status"] == "ACTIVE" and ev["track_id"] == tid:
                ev["temp_end_frame"] = None 
                return eid

        latest_finished_eid = None
        latest_end_frame = -1
        
        for eid, ev in self.events.items():
            if ev["track_id"] == tid and ev["status"] == "FINISHED":
                if ev["end_frame"] > latest_end_frame:
                    latest_end_frame = ev["end_frame"]
                    latest_finished_eid = eid
        
        if latest_finished_eid is not None:
            gap = self.frame_idx - latest_end_frame
            if gap <= self.EVENT_MERGE_GAP:
                ev = self.events[latest_finished_eid]
                ev["status"] = "ACTIVE"
                ev["end_frame"] = None
                ev["temp_end_frame"] = None
                print(f"   🔄 [Merge] Event {latest_finished_eid} resurrected (Gap: {gap} frames)")
                return latest_finished_eid

        eid = next(self.event_id_gen)
        self.events[eid] = {
            "event_id": eid,
            "track_id": tid,
            "start_frame": self.frame_idx,
            "end_frame": None,
            "temp_end_frame": None,
            "status": "ACTIVE",
            "last_dist": dist,
            "last_pos_x": pos_x,
            "warning_type": None,
            "max_decel": 0.0,
            "frame_boxes": {}
        }
        return eid

    def _update_event(self, eid, dist, pos_x, warning_type, acc):
        ev = self.events[eid]
        ev["last_dist"] = dist
        ev["last_pos_x"] = pos_x
        ev["warning_type"] = warning_type
        if acc < ev["max_decel"]:
            ev["max_decel"] = acc

    def _finalize_events(self):
        for ev in self.events.values():
            if ev["status"] != "ACTIVE":
                continue

            tid = ev["track_id"]

            is_warning_stopped = (tid not in self.tracks) or (self.tracks[tid]["warning_cnt"] == 0)

            if is_warning_stopped:
                if ev.get("temp_end_frame") is None:
                    ev["temp_end_frame"] = self.frame_idx
                
                gap = self.frame_idx - ev["temp_end_frame"]
                
                if gap >= self.EVENT_MERGE_GAP:
                    ev["status"] = "FINISHED"
                    ev["end_frame"] = ev["temp_end_frame"]
            else:
                ev["temp_end_frame"] = None

    def _cleanup_tracks(self, current_ids):
        current_ids_set = set(current_ids)
        lost_ids = [tid for tid in self.tracks if tid not in current_ids_set]
        for tid in lost_ids:
            del self.tracks[tid]

    def update(self, boxes, img_w, img_h):
        self.frame_idx += 1
        braking_events = []
        current_ids = []

        self.frame_history[self.frame_idx] = []

        if boxes is None or getattr(boxes, "id", None) is None:
            self._finalize_events()
            self._cleanup_tracks([])
            return braking_events

        for box in boxes:
            if box.id is None or int(box.cls.item()) != 0:
                continue

            tid = int(box.id.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            raw_w_px = float(x2 - x1)
            raw_h_px = float(y2 - y1)
            if raw_w_px < 20:
                continue

            cx = (x1 + x2) / 2.0
            aspect_ratio = raw_w_px / max(raw_h_px, 1.0)

            final_w_px = raw_w_px
            if tid in self.tracks:
                prev_w = self.tracks[tid]["width"]
                final_w_px = prev_w * (1 - self.WIDTH_ALPHA) + raw_w_px * self.WIDTH_ALPHA

            dist = (cfg.FOCAL_LENGTH * cfg.CAR_REAL_WIDTH) / max(final_w_px, 1e-6)
            if dist > 80.0:
                continue

            # frame history 기록
            self.frame_history[self.frame_idx].append({
                "tid": tid,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "dist": dist
            })

            offset_px = cx - (img_w / 2.0)
            pos_x = (offset_px / max(final_w_px, 1e-6)) * cfg.CAR_REAL_WIDTH
            current_ids.append(tid)

            if tid not in self.tracks:
                self._init_track(tid, dist, pos_x, final_w_px, aspect_ratio)
                continue

            prev = self.tracks[tid]

            raw_speed = (dist - prev["dist"]) / self.dt
            if abs(raw_speed) > self.MAX_RAW_SPEED:
                raw_speed = prev["speed"]

            smooth_speed = (
                self.ALPHA_SPEED * raw_speed
                + (1 - self.ALPHA_SPEED) * prev["speed"]
            )

            raw_acc_val = (smooth_speed - prev["speed"]) / self.dt
            if abs(raw_acc_val) > self.MAX_RAW_ACC:
                raw_acc_val = 0.0

            dist_factor = max(0.0, dist - 10.0)
            sensitivity = max(0.4, 1.0 - dist_factor / 60.0)
            raw_acc = raw_acc_val * sensitivity

            ttc = abs(dist / smooth_speed) if smooth_speed < -0.1 else 99.9

            smooth_acc = (
                self.ALPHA_ACC * raw_acc
                + (1 - self.ALPHA_ACC) * prev["acc"]
            )

            lateral_speed = (pos_x - prev["pos_x"]) / self.dt

            if abs(lateral_speed) > self.LATERAL_SKIP_TH:
                smooth_acc = max(smooth_acc, -0.5)
            if dist > self.LONG_RANGE_TH:
                smooth_acc = max(smooth_acc, -0.5)
            if abs(smooth_speed) < 1.0:
                smooth_acc = 0.0

            prev.update({
                "dist": dist,
                "pos_x": pos_x,
                "speed": smooth_speed,
                "acc": smooth_acc,
                "raw_acc": raw_acc,
                "ttc": ttc,
                "width": final_w_px,
                "aspect_ratio": aspect_ratio,
                "cnt": prev["cnt"] + 1,
                "lateral_speed": lateral_speed
            })

            in_lane = abs(pos_x) <= 1.8
            is_stable = prev["cnt"] > 3
            is_cut_in = (
                not in_lane
                and abs(pos_x) < self.CUTIN_WIDTH
                and ((pos_x > 0 and lateral_speed < -0.1) or
                     (pos_x < 0 and lateral_speed > 0.1))
            )

            target_valid = (in_lane or is_cut_in) and is_stable
            is_approaching = smooth_speed < -0.5
            is_not_crawling = smooth_speed < -5.5

            detected_type = cfg.WARN_NORMAL

            if target_valid and is_approaching and is_not_crawling:
                risk = self._calculate_ai_score(ttc, smooth_acc, smooth_speed)
                if risk > 0 and smooth_acc < -1.0:
                    if is_cut_in:
                        detected_type = cfg.WARN_CUT_IN
                    elif ttc < 1.5 and dist < 30.0:
                        detected_type = cfg.WARN_TTC_CRITICAL
                    else:
                        detected_type = cfg.WARN_HARD_BRAKE

            if detected_type != cfg.WARN_NORMAL:
                inc = 2 if detected_type == cfg.WARN_TTC_CRITICAL else 1
                prev["warning_cnt"] += inc
                prev["warning_type"] = detected_type
            else:
                if smooth_acc > 1.0 or ttc > 3.0:
                    prev["warning_cnt"] = 0
                else:
                    prev["warning_cnt"] = max(0, prev["warning_cnt"] - 1)

            if prev["warning_cnt"] > 0:
                eid = self._get_or_create_event(tid, dist, pos_x)
                self._update_event(eid, dist, pos_x, detected_type, smooth_acc)
                prev["last_event_id"] = eid

                if prev["warning_cnt"] >= self.WARNING_DURATION_TH:
                    if self.events[eid] not in braking_events:
                        braking_events.append(self.events[eid])
            else:
                prev["last_event_id"] = None

        self._finalize_events()
        self._cleanup_tracks(current_ids)

        old_frame = self.frame_idx - self.history_limit
        if old_frame in self.frame_history:
            del self.frame_history[old_frame]

        return braking_events
