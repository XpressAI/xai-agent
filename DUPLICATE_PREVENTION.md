# Duplicate Tool Call Prevention

This feature prevents agents from calling the same tool multiple times in a row with identical inputs, which is a common failure mode in frontier models like GPT-5 and Gemini-Flash-2.5.

## How It Works

The system tracks tool calls within each `AgentRun` execution and blocks duplicates based on:
- Tool name
- Normalized arguments (excluding special parameters)

## Special Parameters

### `_duplicate_reasoning`
Allows agents to justify why they need to call the same tool again with the same arguments.

**JSON Format:**
```json
{
  "query": "search term",
  "_duplicate_reasoning": "trying with broader context"
}
```

**XML Format:**
```xml
<query>search term</query>
<_duplicate_reasoning>trying with broader context</_duplicate_reasoning>
```

### `_nonce` 
Reserved for future use. Currently ignored in duplicate detection but preserved in tool arguments.

## Behavior

### ✅ Allowed Cases
1. **First call** - Any tool call is allowed initially
2. **Different tool** - Calling a different tool is always allowed
3. **Different arguments** - Same tool with different parameters is allowed
4. **Unique reasoning** - Duplicate call with unique `_duplicate_reasoning`

### ❌ Blocked Cases
1. **Exact duplicate** - Same tool + same arguments without reasoning
2. **With nonce only** - Same tool + same arguments + `_nonce` but no reasoning
3. **Reused reasoning** - Same tool + same arguments + previously used reasoning

## Examples

```python
# First call - allowed
<tool name="lookup_memory">{"query": "project timeline"}</tool>

# Exact duplicate - blocked
<tool name="lookup_memory">{"query": "project timeline"}</tool>
# Result: ERROR: Duplicate tool call detected...

# With unique reasoning - allowed  
<tool name="lookup_memory">{"query": "project timeline", "_duplicate_reasoning": "searching with broader context"}</tool>

# Reused reasoning - blocked
<tool name="lookup_memory">{"query": "project timeline", "_duplicate_reasoning": "searching with broader context"}</tool>
# Result: ERROR: Duplicate tool call detected for 'lookup_memory' with previously used reasoning...

# Different arguments - allowed
<tool name="lookup_memory">{"query": "project milestones"}</tool>
```

## Error Messages

When duplicates are blocked, clear error messages guide the agent:

- **No reasoning:** "Duplicate tool call detected for 'TOOL_NAME' with same arguments. To retry this tool, provide '_duplicate_reasoning' parameter with unique justification."

- **Reused reasoning:** "Duplicate tool call detected for 'TOOL_NAME' with previously used reasoning: 'REASONING'. Please provide unique justification."

## Implementation Details

- Tracking scope: Within single `AgentRun` execution (resets each run)
- Argument normalization: Removes `_nonce` and `_duplicate_reasoning` for comparison
- Reasoning comparison: Whitespace normalized to prevent trivial bypasses
- Stress level: Increases when duplicates are blocked to encourage different behavior
- Format support: Both JSON and XML argument formats
- Components: Works with both `AgentRun` and `AgentLearn`

## Benefits

🎯 **Prevents infinite loops** - Stops frontier models from getting stuck  
🧠 **Encourages thoughtful usage** - Agents must justify repeat calls  
🔄 **Allows justified retries** - Supports legitimate retry scenarios  
📝 **Provides clear feedback** - Error messages guide better behavior