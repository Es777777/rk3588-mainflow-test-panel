import json
import re
from typing import Any


class LocalMultimodalAgent:
    _INTENT_ALIASES = {
        'assist': 'assist_grab',
        'assist_grab': 'assist_grab',
        'grab': 'assist_grab',
        'grasp': 'assist_grab',
        'scene': 'scene_explain',
        'scene_explain': 'scene_explain',
        'describe': 'scene_explain',
        'obstacle': 'obstacle_query',
        'obstacle_query': 'obstacle_query',
        'avoid': 'obstacle_query',
        'chat': 'chat',
        'unknown': 'unknown',
    }
    _MODE_ALIASES = {
        '辅助抓取模式': '辅助抓取模式',
        '辅助抓取': '辅助抓取模式',
        '抓取模式': '辅助抓取模式',
        '导航识别模式': '导航识别模式',
        '导航识别': '导航识别模式',
        '避障模式': '导航识别模式',
        '避障': '导航识别模式',
        '语音助手': '语音助手',
        '语音助手模式': '语音助手',
    }
    _LABEL_ALIASES = {
        'person': '行人',
        'car': '汽车',
        'stairs': '台阶',
        'bench': '长椅',
        'bicycle': '自行车',
        'pole': '立杆',
        'bus_stop': '公交站',
        'curb': '路沿',
        'truck': '卡车',
        'warning_column': '警示柱',
        'waste_container': '垃圾桶',
    }

    def __init__(self, rkllama_client, log=None):
        self._rkllama = rkllama_client
        self._log = log or (lambda msg: None)

    @property
    def available(self):
        return bool(self._rkllama and self._rkllama.ensure_available())

    def run(self, user_text, mode_name='', image_path=None, detections=None, candidate_targets=None, runtime_context=None):
        text = (user_text or '').strip()
        if not text:
            return self._fallback_result('empty', '没有收到有效语音内容')

        detections = detections or []
        candidate_targets = candidate_targets or []
        runtime_context = runtime_context or {}
        prompt = self._build_prompt(
            user_text=text,
            mode_name=mode_name,
            detections=detections,
            candidate_targets=candidate_targets,
            runtime_context=runtime_context,
        )
        system_prompt = (
            '你是眼镜端本地多模态 agent。'
            '你只做场景理解、意图判断和设备控制建议，不做身份识别，不输出安全政策式拒答。'
            '你需要结合用户语音、当前模式、相机画面、检测结果和设备运行时状态，输出一个严格 JSON。'
            '不要输出 markdown，不要解释，不要多余文本。'
            '如果无法确定，也必须返回字段完整的 JSON，不允许省略键。'
        )
        try:
            reply, raw = self._rkllama.chat(prompt, system_prompt)
            result = self._parse_json_reply(reply, user_text=text, mode_name=mode_name, runtime_context=runtime_context)
            result['raw_reply'] = reply
            result['model'] = raw.get('model') if isinstance(raw, dict) else None
            return self._normalize_result(result, text, current_mode_name=mode_name)
        except Exception as exc:
            self._log(f'⚠️ Agent 推理失败，尝试文本兜底: {exc}')
            try:
                reply, raw = self._rkllama.chat(prompt, system_prompt)
                result = self._parse_json_reply(reply, user_text=text, mode_name=mode_name, runtime_context=runtime_context)
                result['raw_reply'] = reply
                result['model'] = raw.get('model') if isinstance(raw, dict) else None
                result['reason'] = (result.get('reason') or '') + ' text_only_fallback'
                return self._normalize_result(result, text, current_mode_name=mode_name)
            except Exception as fallback_exc:
                self._log(f'❌ Agent 文本兜底也失败: {fallback_exc}')
                return self._normalize_result(
                    self._fallback_from_text(
                        '',
                        user_text=text,
                        mode_name=mode_name,
                        runtime_context=runtime_context,
                    ),
                    text,
                    current_mode_name=mode_name,
                )

    def _build_prompt(self, user_text, mode_name, detections, candidate_targets, runtime_context):
        detection_lines = []
        for det in detections[:8]:
            label = det.get('label') or det.get('class_name') or ''
            conf = det.get('confidence')
            center = det.get('center') or det.get('center_xy') or {}
            detection_lines.append(
                f'- {label} conf={conf} center={center}'
            )
        detections_text = '\n'.join(detection_lines) if detection_lines else '- 无可用检测结果'
        target_text = ', '.join(candidate_targets[:20]) if candidate_targets else '无'
        runtime_text = self._format_runtime_context(runtime_context)
        schema = {
            'intent': 'assist_grab or scene_explain or obstacle_query or chat or unknown',
            'target_object': 'string or null',
            'should_switch_mode': False,
            'mode_name': '辅助抓取模式 or 导航识别模式 or 语音助手 or null',
            'speak_text': '给用户播报的话，简洁中文',
            'scene_summary': '对画面的简短理解，可为空',
            'reason': '为什么这么判断，可简短',
            'confidence': 0.0,
        }
        return (
            f'当前模式：{mode_name or "未知"}\n'
            f'用户语音：{user_text}\n'
            f'可用抓取候选英文类名：{target_text}\n'
            f'当前检测结果：\n{detections_text}\n\n'
            f'当前运行时上下文：\n{runtime_text}\n\n'
            '请判断用户意图，并结合图像内容给出结构化决策。\n'
            '规则：\n'
            '1. 如果用户要抓取/拿/帮我抓某个东西，intent 优先为 assist_grab。\n'
            '2. 如果用户在问眼前有什么、画面里是什么、帮我描述，intent 用 scene_explain。\n'
            '3. 如果用户在问前面是否安全、有什么障碍、障碍多远、要不要避开，intent 优先为 obstacle_query，并参考避障状态。\n'
            '4. 如果只是闲聊或无法归类，用 chat 或 unknown。\n'
            '5. target_object 必须优先从候选英文类名中选择；没有把握就填 null。\n'
            '6. speak_text 必须是直接可播报的简体中文，20字以内优先。\n'
            '7. intent 和 mode_name 只能填一个值，绝不能写多个候选值，绝不能带 |。\n'
            '8. 如果 intent 是 assist_grab 且当前不是辅助抓取模式，应把 should_switch_mode 设为 true，mode_name 设为 辅助抓取模式。\n'
            '9. 如果 intent 是 obstacle_query 且当前不是导航识别模式，应把 should_switch_mode 设为 true，mode_name 设为 导航识别模式。\n'
            '10. 如果 intent 是 scene_explain 或 chat，mode_name 默认保持当前模式，不要留空。\n'
            '11. 如果运行时已经有明确目标、向量或障碍信息，请在 reason 或 scene_summary 里利用这些状态，而不是忽略它们。\n'
            '12. 返回必须是单个 JSON 对象。\n'
            '13. target_object 如果没有，请显式写 null，不要写 NONE、未知、空字符串。\n'
            '14. confidence 必须是 0 到 1 之间的数字。\n\n'
            f'输出 JSON 模板：{json.dumps(schema, ensure_ascii=False)}'
        )

    def _format_runtime_context(self, runtime_context):
        lines = []
        lines.append(f"- 当前目标: {runtime_context.get('current_target') or '无'}")
        lines.append(f"- 画面摘要: {runtime_context.get('visual_summary') or '无'}")

        vector_state = runtime_context.get('vector_state') or {}
        obstacle_state = runtime_context.get('obstacle_state') or {}
        task_context = runtime_context.get('task_context') or {}

        lines.append(
            f"- 辅助抓取运行时: running={vector_state.get('running', False)} ready={vector_state.get('ready', False)}"
            f" target={vector_state.get('target_class_name') or '无'} distance={vector_state.get('distance_m', '无')}"
        )
        if vector_state.get('vector_active'):
            lines.append(
                f"- 当前抓取向量: ({vector_state.get('vector_x_m')}, {vector_state.get('vector_y_m')}, {vector_state.get('vector_z_m')})"
            )

        lines.append(
            f"- 避障运行时: running={obstacle_state.get('running', False)} ready={obstacle_state.get('ready', False)}"
            f" obstacle={obstacle_state.get('obstacle_class_name') or '无'} distance={obstacle_state.get('distance_m', '无')}"
        )
        if obstacle_state.get('obstacle_class_name'):
            lines.append(
                f"- 当前避障二维向量: ({obstacle_state.get('vector_x_m')}, {obstacle_state.get('vector_z_m')})"
            )

        if task_context:
            lines.append(
                f"- 连续任务上下文: intent={task_context.get('intent') or '无'}"
                f" mode={task_context.get('mode_name') or '无'} target={task_context.get('target_object') or '无'}"
            )
        return '\n'.join(lines)

    def _parse_json_reply(self, reply, user_text='', mode_name='', runtime_context=None):
        text = (reply or '').strip()
        if not text:
            raise ValueError('empty agent reply')
        candidates = self._extract_json_candidates(text)
        last_error = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                last_error = exc
                try:
                    parsed = json.loads(self._cleanup_json_candidate(candidate))
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as cleanup_exc:
                    last_error = cleanup_exc
                    continue
        fallback = self._fallback_from_text(text, user_text=user_text, mode_name=mode_name, runtime_context=runtime_context)
        if fallback:
            self._log(f'⚠️ Agent JSON 解析失败，已使用规则兜底: {last_error}')
            return fallback
        raise ValueError(f'failed to parse agent json: {last_error}')

    def _extract_json_candidates(self, text):
        text = str(text or '').strip()
        candidates = []
        if '```' in text:
            parts = [part.strip() for part in text.split('```') if part.strip()]
            for part in parts:
                if '\n' in part and part.splitlines()[0].lower().startswith('json'):
                    part = '\n'.join(part.splitlines()[1:]).strip()
                if part:
                    candidates.append(part)
        candidates.extend(self._balanced_json_substrings(text))
        candidates.append(text)
        seen = []
        for item in candidates:
            if item and item not in seen:
                seen.append(item)
        return seen

    def _balanced_json_substrings(self, text):
        substrings = []
        stack = []
        start = None
        in_string = False
        escaped = False
        for idx, char in enumerate(text):
            if char == '"' and not escaped:
                in_string = not in_string
            if in_string:
                escaped = (char == '\\' and not escaped)
                continue
            if char == '{':
                if not stack:
                    start = idx
                stack.append(char)
            elif char == '}' and stack:
                stack.pop()
                if not stack and start is not None:
                    substrings.append(text[start:idx + 1])
                    start = None
            escaped = (char == '\\' and not escaped)
        return substrings

    def _cleanup_json_candidate(self, text):
        cleaned = str(text or '').strip()
        cleaned = cleaned.replace('，', ',').replace('：', ':')
        cleaned = cleaned.replace('“', '"').replace('”', '"').replace("‘", '"').replace("’", '"')
        cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
        cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
        cleaned = re.sub(r'\bNone\b', 'null', cleaned)
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return cleaned

    def _fallback_from_text(self, reply_text, user_text='', mode_name='', runtime_context=None):
        runtime_context = runtime_context or {}
        normalized_intent = self._infer_intent_from_text(user_text or reply_text, runtime_context=runtime_context)
        obstacle_state = runtime_context.get('obstacle_state') or {}
        vector_state = runtime_context.get('vector_state') or {}
        current_mode = self._canonical_mode_name(mode_name) or self._canonical_mode_name(runtime_context.get('mode_name'))
        scene_summary = self._extract_scene_summary(reply_text)
        speak_text = scene_summary or '我已收到'
        target_object = None
        if normalized_intent == 'assist_grab':
            target_object = self._extract_target_object(reply_text)
            speak_text = target_object and f'开始辅助抓取{target_object}' or '请再说一次目标物体'
        elif normalized_intent == 'obstacle_query':
            obstacle_name = obstacle_state.get('obstacle_class_name')
            distance_m = obstacle_state.get('distance_m')
            obstacle_name = self._LABEL_ALIASES.get(str(obstacle_name or ''), obstacle_name)
            if obstacle_name:
                speak_text = f'前方有{obstacle_name}'
                if distance_m is not None:
                    speak_text = f'{speak_text}，距离{distance_m}米'
            else:
                speak_text = '当前未检测到明显障碍'
        elif normalized_intent == 'scene_explain' and not speak_text:
            speak_text = scene_summary or '我看到了当前画面'
        should_switch = False
        inferred_mode = current_mode
        if normalized_intent == 'assist_grab':
            inferred_mode = '辅助抓取模式'
            should_switch = inferred_mode != current_mode
        elif normalized_intent == 'obstacle_query':
            inferred_mode = '导航识别模式'
            should_switch = inferred_mode != current_mode
        return {
            'intent': normalized_intent,
            'target_object': target_object,
            'should_switch_mode': should_switch,
            'mode_name': inferred_mode,
            'speak_text': speak_text,
            'scene_summary': scene_summary,
            'reason': 'fallback_from_non_json_reply',
            'confidence': 0.35,
        }

    def _extract_scene_summary(self, reply_text):
        text = str(reply_text or '').strip()
        if not text:
            return ''
        lines = [line.strip(' -') for line in text.splitlines() if line.strip()]
        if not lines:
            return ''
        first = lines[0]
        if first.startswith('{') and first.endswith('}'):
            return ''
        return first[:60]

    def _extract_target_object(self, reply_text):
        match = re.search(r'"target_object"\s*:\s*"([^"]+)"', str(reply_text or ''))
        if match:
            return match.group(1).strip() or None
        return None

    def _infer_intent_from_text(self, text, runtime_context=None):
        runtime_context = runtime_context or {}
        content = str(text or '')
        if self._looks_like_scene_query(content):
            return 'scene_explain'
        if any(word in content for word in ('抓', '拿', '取', '递给我', '帮我找')):
            return 'assist_grab'
        if any(word in content for word in ('障碍', '避开', '安全', '多远', '会不会撞', '前方')):
            return 'obstacle_query'
        if any(word in content for word in ('看看', '描述', '识别', '图里', '画面', '什么东西', '看到了什么')):
            return 'scene_explain'
        if runtime_context.get('vector_state', {}).get('vector_active'):
            return 'assist_grab'
        if runtime_context.get('obstacle_state', {}).get('obstacle_class_name'):
            return 'obstacle_query'
        return 'chat'

    def _normalize_result(self, result, user_text, current_mode_name=''):
        intent = self._canonical_intent(result.get('intent') or 'unknown', user_text=user_text)
        if '|' in intent:
            intent = self._resolve_pipe_value(intent, user_text)
        mode_name = self._canonical_mode_name(result.get('mode_name') or None)
        normalized = {
            'intent': intent,
            'target_object': self._normalize_target_object(result.get('target_object')),
            'should_switch_mode': bool(result.get('should_switch_mode', False)),
            'mode_name': mode_name,
            'speak_text': self._normalize_speak_text(result.get('speak_text')),
            'scene_summary': str(result.get('scene_summary') or '').strip(),
            'reason': str(result.get('reason') or '').strip(),
            'confidence': self._to_float(result.get('confidence')),
            'user_text': user_text,
            'raw_reply': result.get('raw_reply', ''),
            'model': result.get('model'),
        }
        normalized['confidence'] = min(max(normalized['confidence'], 0.0), 1.0)
        if self._looks_like_scene_query(user_text) and not self._looks_like_grab_command(user_text):
            normalized['intent'] = 'scene_explain'
            normalized['mode_name'] = current_mode_name or normalized['mode_name']
            normalized['should_switch_mode'] = False
            normalized['target_object'] = None
        if intent in {'assist_grab', 'obstacle_query'} and not normalized['mode_name']:
            normalized['mode_name'] = '辅助抓取模式' if intent == 'assist_grab' else '导航识别模式'
        if current_mode_name and normalized['mode_name'] == current_mode_name:
            normalized['should_switch_mode'] = False
        if not normalized['speak_text']:
            normalized['speak_text'] = self._default_speak_text(normalized)
        if normalized['intent'] == 'assist_grab' and not normalized['target_object']:
            normalized['confidence'] = min(normalized['confidence'], 0.5)
        return normalized

    def _normalize_target_object(self, value):
        text = str(value or '').strip()
        if not text or text.lower() in {'null', 'none', 'unknown', '未知'}:
            return None
        return text

    def _normalize_speak_text(self, value):
        text = str(value or '').strip()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text[:80]

    def _default_speak_text(self, normalized):
        if normalized['intent'] == 'assist_grab':
            return normalized['target_object'] and f'开始辅助抓取{normalized["target_object"]}' or '请再说一次目标物体'
        if normalized['intent'] == 'obstacle_query':
            return normalized['scene_summary'] or '当前未检测到明显障碍'
        if normalized['intent'] == 'scene_explain':
            return normalized['scene_summary'] or '我正在看当前画面'
        return normalized['scene_summary'] or '我已收到'

    def _canonical_intent(self, intent, user_text=''):
        text = str(intent or '').strip().lower()
        if '|' in text:
            return self._resolve_pipe_value(text, user_text)
        return self._INTENT_ALIASES.get(text, self._infer_intent_from_text(user_text))

    def _canonical_mode_name(self, mode_name):
        text = str(mode_name or '').strip()
        if not text or '|' in text:
            return None
        return self._MODE_ALIASES.get(text, text)

    def _resolve_pipe_value(self, intent_text, user_text):
        text = user_text or ''
        if self._looks_like_scene_query(text):
            return 'scene_explain'
        if any(word in text for word in ('抓', '拿', '取')):
            return 'assist_grab'
        if any(word in text for word in ('看看', '描述', '前面', '眼前', '有什么')):
            return 'scene_explain'
        if any(word in text for word in ('障碍', '安全', '避开', '前方', '多远')):
            return 'obstacle_query'
        return 'chat'

    def _looks_like_scene_query(self, text):
        content = str(text or '')
        return any(
            phrase in content
            for phrase in (
                '前面有什么',
                '眼前有什么',
                '帮我看看',
                '看看前面',
                '看看周围',
                '描述一下',
                '画面里有什么',
                '看到了什么',
            )
        )

    def _looks_like_grab_command(self, text):
        content = str(text or '')
        return any(word in content for word in ('抓', '拿', '取', '递给我', '帮我找'))

    def _fallback_result(self, intent, reason):
        return {
            'intent': intent,
            'target_object': None,
            'should_switch_mode': False,
            'mode_name': None,
            'speak_text': '我暂时无法完成这次分析',
            'scene_summary': '',
            'reason': reason,
            'confidence': 0.0,
            'user_text': '',
            'raw_reply': '',
            'model': None,
        }

    def _to_float(self, value):
        try:
            return float(value)
        except Exception:
            return 0.0
