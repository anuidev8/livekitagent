# Agent Builder setup (editable in LiveKit Cloud UI)

Use this only if the experience runs from a LiveKit Agent Builder agent. The Python code agent in this repository already exposes the same semantic tool contract.

## 1. Create or open Agent Builder

1. Open the Agents area of the intended LiveKit Cloud project.
2. Create a Builder agent, or open the existing one.
3. Use a distinct agent name while testing so it does not compete with the deployed Python agent.

## 2. Paste the instructions

Copy all of `AGENT_BUILDER_INSTRUCTIONS.txt` into the Builder **Instructions** field.

## 3. Add four client tools

Names and parameter shapes must match exactly. The frontend validates actions against the current screen, so no phrase-to-command mapping is needed.

### `get_session_state`

- Parameters: none
- Purpose: read the current screen, phase, profile, available actions, UI focus, scores, and exact visible content. Call it first on every turn.
- Example result:

```json
{
  "step": "welcome",
  "phase": "ready",
  "availableActions": ["advance", "back"],
  "profile": {
    "name": "Nombre dinámico",
    "role": "Cargo dinámico",
    "company": "Empresa dinámica"
  }
}
```

### `navigate_journey`

- Parameters:
  - `action` (required string): `advance`, `back`, `start_experience`, `start_analysis`, `open_detail`, `send_report`, `skip_report`, `ready_for_picture`, `capture_picture`, `finish`, or `cancel`
  - `dimensionId` (optional string): dimension to open when the action is `open_detail`
- Purpose: perform a semantic navigation or confirmation after the user's intent is clear.
- Example result:

```json
{"ok":true,"action":"open_detail","step":"detail","dimensionId":"authority"}
```

### `present_content`

- Parameters:
  - `target` (required string): `attract_tour`, `welcome_preparation`, `intro_step`, `intro_dimension`, `result_dimension`, `detail_dimension`, or `detail_section`
  - `index` (optional number): visible card or dimension index
  - `dimensionId` (optional string): dimension identifier
  - `section` (optional string): `strengths`, `opportunities`, or `action_plan`
- Purpose: focus exactly one visible item before narrating it. This keeps the UI and voice synchronized.
- Example result:

```json
{
  "ok": true,
  "target": "detail_section",
  "dimensionId": "authority",
  "section": "opportunities"
}
```

### `set_control_channel`

- Parameters:
  - `channel` (required string): `voice` or `gesture`
  - `enabled` (required boolean)
- Purpose: change an input channel only when the user requests it.
- Example result:

```json
{"ok":true,"channel":"gesture","enabled":false}
```

## 4. Deploy and point the kiosk

Deploy the Builder agent, then set the exact deployed agent name in `huella-digital/.env.local`:

```dotenv
LIVEKIT_AGENT_NAME=<deployed-agent-name>
```

Restart the Next.js application after changing the environment value. Do not run two agents with the same dispatch target during validation.
