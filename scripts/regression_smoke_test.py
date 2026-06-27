import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.local_multimodal_agent import LocalMultimodalAgent
from app.main import AppState


def build_state():
    state = AppState()
    state.modes = [
        SimpleNamespace(name='辅助抓取模式', on_enter=lambda: None, on_exit=lambda: None),
        SimpleNamespace(name='导航识别模式', on_enter=lambda: None, on_exit=lambda: None),
        SimpleNamespace(name='语音助手', on_enter=lambda: None, on_exit=lambda: None),
    ]
    return state


def test_obstacle_speech():
    state = build_state()
    obstacle = {'label': 'person', 'distance_m': 0.86, 'vector_x_m': -0.12, 'vector_z_m': 0.84}
    speech = state._build_obstacle_speech(obstacle)
    assert speech == '左前方有行人，距离0.86米', speech

    obstacle = {'label': 'stairs', 'distance_m': 1.43, 'vector_x_m': 0.02, 'vector_z_m': 1.43}
    speech = state._build_obstacle_speech(obstacle)
    assert speech == '正前方有台阶，距离1.43米', speech


def test_wristband_packets():
    state = build_state()
    state.set_mode(0)
    state.set_vector_payload({
        'frame_index': 3,
        'status': 'tracking',
        'vector_active': True,
        'vector_x_m': 0.1,
        'vector_y_m': -0.03,
        'vector_z_m': 0.02,
        'distance_m': 0.15,
    })
    packet = state._build_grasp_wristband_packet(state.vector_payload)
    assert packet['mode'] == 'grasp'
    assert packet['vector_active'] is True

    state.set_mode(1)
    state.set_obstacle_payload({
        'frame_index': 6,
        'obstacle_class_name': 'person',
        'vector_x_m': -0.12,
        'vector_z_m': 0.84,
        'distance_m': 0.86,
    })
    packet = state._build_avoid_wristband_packet(state.obstacle_payload)
    assert packet['mode'] == 'avoid'
    assert packet['vector_y_m'] == 0.0


def test_agent_parser():
    agent = LocalMultimodalAgent(None)
    reply = '''```json
{
  "intent": "obstacle_query",
  "target_object": null,
  "should_switch_mode": false,
  "mode_name": "导航识别模式",
  "speak_text": "左前方有行人，距离0.86米",
  "scene_summary": "左前方有行人",
  "reason": "检测到前方障碍",
  "confidence": 0.82,
}
```'''
    parsed = agent._parse_json_reply(reply, user_text='前面安全吗', mode_name='导航识别模式', runtime_context={})
    assert parsed['intent'] == 'obstacle_query'
    assert parsed['mode_name'] == '导航识别模式'

    fallback = agent._parse_json_reply(
        '前面有障碍，建议避开，当前看到左前方有行人。',
        user_text='前面安全吗',
        mode_name='导航识别模式',
        runtime_context={'obstacle_state': {'obstacle_class_name': 'person', 'distance_m': 0.86}},
    )
    assert fallback['intent'] == 'obstacle_query'
    assert fallback['speak_text']


def main():
    test_obstacle_speech()
    test_wristband_packets()
    test_agent_parser()
    print('regression_smoke_test: OK')


if __name__ == '__main__':
    main()
