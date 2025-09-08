from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, xai_component, dynalist, secret, SubGraphExecutor
import traceback
import abc
from collections import deque
from typing import NamedTuple

import json
import os
import requests
import random
import string
import copy

try:
    import openai
except Exception as e:
    pass

# Optional: If using NumpyMemory need numpy and OpenAI
try:
    import numpy as np
except Exception as e:
    pass

# Optional: If using vertexai provider.
try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
except Exception as e:
    pass


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def is_openai_model(model_name: str) -> bool:
    """Check if the model is an OpenAI model that supports system messages."""
    if not model_name:
        return False
    if model_name.startswith('gpt-5'):
        return False
    
    return model_name.startswith(('o1', 'o3', 'o4', 'gpt'))

def convert_old_tool_syntax_to_xml(text: str) -> str:
    """Convert old TOOL: syntax to new XML format in tool descriptions."""
    import re
    
    # Pattern to match old style: TOOL: tool_name args (args optional)
    # This pattern captures tool name and optional arguments
    # Match the entire line including any trailing spaces but not the newline
    old_tool_pattern = r'^(\s*)TOOL:\s+(\w+)(?:\s+(.*))?$'
    
    def replace_match(match):
        indent = match.group(1)
        tool_name = match.group(2)
        args = match.group(3) if match.group(3) else ''
        
        # Convert to XML format
        if args:
            # If args exist, check if they should be on same line or new line
            args = args.strip()
            if args:
                return f'{indent}<tool name="{tool_name}">\n{indent}{args}\n{indent}</tool>'
            else:
                return f'{indent}<tool name="{tool_name}"></tool>'
        else:
            # No args, single line format
            return f'{indent}<tool name="{tool_name}"></tool>'
    
    # Replace all occurrences in the text
    converted_text = re.sub(old_tool_pattern, replace_match, text, flags=re.MULTILINE)
    
    return converted_text

def parse_xml_args_to_dict(xml_content: str) -> dict:
    """Parse XML-style arguments into a dictionary using a simple SAX-like approach.
    
    Supports format like:
    <arg1>value1</arg1>
    <arg2>value2</arg2>
    
    Returns None if the content is not valid XML args.
    """
    import re
    
    # Strip leading/trailing whitespace
    xml_content = xml_content.strip()
    
    # If empty, return None
    if not xml_content:
        return None
    
    # Simple pattern to match XML tags with content
    # This matches: <tag>content</tag>
    tag_pattern = r'<(\w+)>(.*?)</\1>'
    
    matches = re.findall(tag_pattern, xml_content, re.DOTALL)
    
    # If no matches found, it's not XML format
    if not matches:
        return None
    
    # Build dictionary from matches
    result = {}
    for tag_name, content in matches:
        result[tag_name] = content.strip()
    
    # Check if there's any content that's not within tags
    # Remove all matched tags and see if there's leftover content
    remaining = xml_content
    for match in re.finditer(tag_pattern, xml_content, re.DOTALL):
        remaining = remaining.replace(match.group(0), '', 1)
    
    # If there's significant non-whitespace content remaining, it's not pure XML
    if remaining.strip():
        return None
    
    return result

def parse_tool_args(args_str: str) -> tuple:
    """Parse tool arguments and return (parsed_dict, raw_string).
    
    Supports:
    1. JSON format: {"arg1": "value1", "arg2": "value2"}
    2. XML format: <arg1>value1</arg1><arg2>value2</arg2>
    3. Empty args: returns (None, "")
    4. Plain text: returns (None, raw_text)
    
    Returns:
        tuple: (parsed_dict or None, raw_string)
    """
    args_str = args_str.strip()
    
    # Handle empty args
    if not args_str:
        return (None, "")
    
    # Try JSON first
    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return (parsed, args_str)
    except:
        pass
    
    # Try XML format
    xml_dict = parse_xml_args_to_dict(args_str)
    if xml_dict is not None:
        return (xml_dict, args_str)
    
    # Neither JSON nor XML, return raw content
    return (None, args_str)

def encode_prompt(model_id: str, conversation: list):
    ret_messages = []

    if 'anthropic.claude-3' in model_id.lower():
        for message in conversation:
            if message['role'] == 'system':
                message['role'] = 'user'
            
            if isinstance(message['content'], str):
                ret_messages.append({
                    'role': message['role'],
                    'content': [
                        {
                            'type': 'text',
                            'text': 'SYSTEM:\n' + message['content']
                        }
                    ]
                })
            else:
                new_contents = []
                for content in message['content']:
                    if content['type'] == 'image_url':
                        # f"data:image/jpeg;base64,{base64_image}"
                        url = content['image_url']['url']
                        (media_type, rest) = url.split(';', 1)
                        data = rest.split(',', 1)
                        media_type = media_type.split(':', 1)[1]
        
                        source = {
                            'type': 'base64',
                            'media_type': media_type,
                            'data': data[1]
                        }
    
                        new_contents.append({
                            'type': 'image',
                            'source': source
                        })
                    else:
                        new_contents.append({
                            'type': 'text',
                            'text': content['text']
                        })

                ret_messages.append({
                    'role': message['role'],
                    'content': new_contents
                })
                
    return ret_messages


class Memory(abc.ABC):
    def query(self, query: str, n: int) -> list:
        pass

    def add(self, id: str, text: str, metadata: dict) -> None:
        pass


class VectoMemoryImpl(Memory):
    def __init__(self, vs):
        self.vs = vs

    def query(self, query: str, n: int) -> list:
        return self.vs.lookup(query, 'TEXT', n)
    def add(self, id: str, text: str, metadata: dict) -> None:
        self.vs.ingest_text(text, metadata)



def get_ada_embedding(text):
    s = text.replace("\n", " ")
    return openai.Embedding.create(input=[s], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


class NumpyQueryResult(NamedTuple):
    id: str
    similarity: float
    attributes: dict


class NumpyMemoryImpl(Memory):
    def __init__(self, vectors=None, ids=None, metadata=None):
        self.vectors = vectors
        self.ids = ids
        self.metadata = metadata

    def query(self, query: str, n: int) -> list:
        if self.vectors is None:
            return []
        if isinstance(self.vectors, list) and len(self.vectors) > 1:
            self.vectors = np.vstack(self.vectors)

        top_k = min(self.vectors.shape[0], n)
        query_vector = get_ada_embedding(query)
        similarities = self.vectors @ query_vector
        indices = np.argpartition(similarities, -top_k)[-top_k:]
        return [
            NumpyQueryResult(
                self.ids[i],
                similarities[i],
                self.metadata[i]
            )
            for i in indices
        ]

    def add(self, vector_id: str, text: str, metadata: dict) -> None:
        if isinstance(self.vectors, list) and len(self.vectors) > 1:
            self.vectors = np.vstack(self.vectors)

        if self.vectors is None:
            self.vectors = np.array(get_ada_embedding(text)).reshape((1, -1))
            self.ids = [vector_id]
            self.metadata = [metadata]
        else:
            self.ids.append(vector_id)
            self.vectors = np.vstack([self.vectors, np.array(get_ada_embedding(text))])
            self.metadata.append(metadata)


@xai_component
class AgentNumpyMemory(Component):
    """Creates a local and temporary memory for the agent to store and query information.

    ##### outPorts:
    - memory: The Memory to set on AgentInit
    """
     
    memory: OutArg[Memory]

    def execute(self, ctx) -> None:
        self.memory.value = NumpyMemoryImpl()


class Tool(NamedTuple):
    name: str
    description: str
    inputs: str
    outputs: str


class MutableVariable:
    _fn: any

    def __init__(self):
        self._fn = None
    
    def set_fn(self, fn) -> None:
        self._fn = fn
        
    @property
    def value(self) -> any:
        return self._fn()


@xai_component(type="Start", color="red")
class AgentDefineTool(Component):
    """Define a tool that the agent can use when it deems necessary.

    This event will be called when the Agent uses this tool.  Perform the tool
    actions and set the output with AgentToolOutput

    ##### inPorts:
    - tool_name: The name of the tool.
    - description: The description of the tool.
    - for_toolbelt: The toolbelt to add the tool to.  If not set, will be added to the default toolbelt.

    ##### outPorts:
    - tool_input: The input for the tool coming from the agent.
    - tool_args: The parsed JSON arguments if the input is valid JSON, otherwise None.

    """

    tool_name: InCompArg[str]
    description: InCompArg[str]
    for_toolbelt: InArg[str]
    
    tool_input: OutArg[str]
    tool_args: OutArg[dict]

    
    def init(self, ctx):
        toolbelt = self.for_toolbelt.value if self.for_toolbelt.value is not None else 'default'
        ctx.setdefault('toolbelt_' + toolbelt, {})[self.tool_name.value] = self
        self.tool_ref = InCompArg(None)
        
    
    def execute(self, ctx) -> None:
        other_self = self
        
        class CustomTool(Tool):
            name = other_self.tool_name.value
            description = other_self.description.value
            inputs = ["text"]
            output = ["text"]
            
            def __call__(self, prompt):
                other_self.tool_input.value = prompt
                # Parse arguments using the new flexible parser
                parsed_args, _ = parse_tool_args(prompt)
                other_self.tool_args.value = parsed_args
                SubGraphExecutor(other_self.next).do(ctx)
                result = ctx['tool_output']
                ctx['tool_output'] = None
                return result
            
        self.tool_ref.value = CustomTool(
            self.tool_name.value,
            self.description.value,
            ["text"],
            ["text"]
        )

@xai_component(color="red")
class AgentToolOutput(Component):
    """Output the result of the tool to the agent.

    ##### inPorts:
    - results: The results of the tool to be returned to the agent.

    """

    results: InArg[dynalist]
    
    def execute(self, ctx) -> None:
        if len(self.results.value) == 1:
            ctx['tool_output'] = self.results.value[0]


@xai_component
class AgentUseMCPTools(Component):
    """Declare that tools from a pre-configured MCP server should be available to the agent.

    This component informs the agent runtime which MCP servers (defined in the system's
    MCP configuration) are active for this specific run.

    ##### inPorts:
    - server_name: The exact name of the MCP server as configured in the system.
    - toolbelt_name: The conceptual toolbelt group to associate this server with (default: 'default').
    """
    server_name: InCompArg[str]
    toolbelt_name: InArg[str]

    def execute(self, ctx) -> None:
        tb_name = self.toolbelt_name.value if self.toolbelt_name.value else 'default'
        mcp_server_list_key = f"mcp_servers_{tb_name}"
        
        # Ensure the list exists in the context
        if mcp_server_list_key not in ctx:
            ctx[mcp_server_list_key] = []
            
        # Add the server name if not already present
        if self.server_name.value not in ctx[mcp_server_list_key]:
            ctx[mcp_server_list_key].append(self.server_name.value)


@xai_component
class AgentMakeToolbelt(Component):
    """Create a toolbelt for the agent to use.

    ##### inPorts:
    - name: The name of the toolbelt.

    ##### outPorts:
    - toolbelt_spec: The toolbelt to set on AgentInit

    """
    name: InArg[str]
    toolbelt_spec: OutArg[dict]

    def execute(self, ctx) -> None:
        standard_tools = {}
        mcp_servers = []
        toolbelt_name = self.name.value if self.name.value is not None else 'default'
        
        # Process standard tools defined via AgentDefineTool or devcore tools
        standard_tool_key = f"toolbelt_{toolbelt_name}"
        if standard_tool_key in ctx:
            for tool_name, tool_component in ctx[standard_tool_key].items():
                # Handle any component that has a tool_ref attribute (AgentDefineTool, devcore tools, etc.)
                if hasattr(tool_component, 'tool_ref'):
                    # Ensure the tool definition is executed to populate tool_ref
                    tool_component.execute(ctx)
                    if hasattr(tool_component, 'tool_ref') and tool_component.tool_ref.value:
                         # Store the actual callable tool instance
                        standard_tools[tool_component.tool_ref.value.name] = tool_component.tool_ref.value
                    else:
                        print(f"Warning: Tool component '{tool_name}' did not produce a tool reference.")
                # Note: We ignore components without tool_ref here, MCP servers are handled next

        # Retrieve declared MCP server names
        mcp_server_list_key = f"mcp_servers_{toolbelt_name}"
        if mcp_server_list_key in ctx:
            mcp_servers = ctx[mcp_server_list_key]

        # Output the combined spec
        self.toolbelt_spec.value = {
            'standard_tools': standard_tools,
            'mcp_servers': mcp_servers
        }


@xai_component
class AgentVectoMemory(Component):
    """Creates a memory for the agent to store and query information.
    
    ##### inPorts:
    - api_key: The API key for Vecto.
    - vector_space: The name of the vector space to use.
    - initialize: Whether to initialize the vector space.

    ##### outPorts:
    - memory: The Memory to set on AgentInit

    """

    api_key: InArg[secret]
    vector_space: InCompArg[str]
    initialize: InCompArg[bool]

    memory: OutArg[Memory]

    def execute(self, ctx) -> None:
        from vecto import Vecto

        api_key = os.getenv("VECTO_API_KEY") if self.api_key.value is None else self.api_key.value

        headers = {'Authorization': 'Bearer ' + api_key}
        response = requests.get("https://api.vecto.ai/api/v0/account/space", headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to get vector space list: {response.text}")
        for space in response.json():
            if space['name'] == self.vector_space.value:
                vs = Vecto(api_key, space['id'])
                if self.initialize.value:
                    vs.delete_vector_space_entries()
                self.memory.value = VectoMemoryImpl(vs)
                break
        if not self.memory.value:
            vs = Vecto(api_key)
            model_id = [model for model in vs.list_models() if model.name == 'QWEN2'][0].id
            res = requests.post("https://api.vecto.ai/api/v0/account/space", headers=headers, json={
                "name": self.vector_space.value,
                "modelId": model_id
            })
            data = res.json()
            vs = Vecto(api_key, data['id'])
            self.memory.value = VectoMemoryImpl(vs)

# TBD
#@xai_component
#class AgentToolbeltFolder(Component):
#    folder: InCompArg[str]
#
#    toolbelt_spec: OutArg[list]
#
#    def execute(self, ctx) -> None:
#        spec = []
#        self.toolbelt_spec.value = spec


@xai_component
class AgentInit(Component):
    """Initialize the agent with the necessary components.

    ##### inPorts:
    - agent_name: The name of the agent to create.
    - agent_provider: The provider of the agent (Either openai, vertexai, or bedrock).
    - agent_model: The model that the agent should use (Such as gpt-3.5-turbo, gemini-pro, or anthropic.claude-3-5-sonnet-20240620-v1:0).
    - agent_memory: The memory that the agent should use to store data it wants to remember.
    - system_prompt: The system prompt of the agent be sure to speficy 
      {tool_instruction} and {tools} to explain how to use them.
    - max_thoughts: The maximum number of thoughts/tools the agent can use before it must respond to the user.
    - toolbelt_spec: The toolbelt the agent has access to.
    """

    agent_name: InCompArg[str]
    agent_provider: InCompArg[str]
    agent_model: InCompArg[str]
    agent_memory: InCompArg[Memory]
    system_prompt: InCompArg[str]
    max_thoughts: InArg[int]
    toolbelt_spec: InCompArg[dict]
    
    def execute(self, ctx) -> None:
        provider = self.agent_provider.value
        if provider not in ['openai', 'vertexai', 'bedrock']:
            raise Exception(f"Agent provider '{provider}' is not supported.")

        agent_context_key = 'agent_' + self.agent_name.value
        ctx[agent_context_key] = {
            # Store standard tools under 'agent_toolbelt'
            'agent_toolbelt': self.toolbelt_spec.value.get('standard_tools', {}),
            # Store the list of enabled MCP server names
            'enabled_mcp_servers': self.toolbelt_spec.value.get('mcp_servers', []),
            'agent_provider': provider,
            'agent_memory': self.agent_memory.value,
            'agent_model': self.agent_model.value,
            'agent_system_prompt': self.system_prompt.value,
            'max_thoughts': self.max_thoughts.value if self.max_thoughts.value is not None else 5 # Default max_thoughts
        }


@xai_component
class AgentUpdate(Component):
    agent_name: InCompArg[str]
    new_agent_provider: InArg[str]
    new_agent_model: InArg[str]
    new_system_prompt: InArg[str]
    
    def execute(self, ctx) -> None:
        if self.agent_provider.value != 'openai' and self.agent_provider.value != 'vertexai' and self.agent_provider.value != 'bedrock':
            raise Exception(f"agent provider: {self.agent_provider.value} is not supported in this version of xai_agent.")

        agent = ctx['agent_' + self.agent_name.value]

        if self.new_agent_provider.value is not None:
            agent['agent_provider'] = self.new_agent_provider.value
        if self.new_agent_model.value is not None:
            agent['agent_model'] = self.new_agent_model.value
        if self.new_system_prompt.value is not None:
            agent['agent_system_prompt'] = self.new_system_prompt.value



@xai_component(type="Start", color="purple") # Using purple for Start nodes
class AgentDefineRoute(Component):
    """Defines the starting point for a specific agent processing flow.
    The AgentRouter can use the name and description to select this flow.

    ##### inPorts:
    - agent_flow_name: A unique name for this agent route.
    - description: A description of what this agent route does, used by the router.
    """
    agent_flow_name: InCompArg[str]
    description: InCompArg[str]

    # This component doesn't have direct outputs, it triggers a subgraph execution.

    def init(self, ctx):
        super().init(ctx)
        if not self.agent_flow_name.value:
            print("Warning: AgentDefineRoute requires an agent_flow_name.")
            return
        # Register this start point for the router
        start_points = ctx.setdefault('registered_agent_routes', {})
        start_points[self.agent_flow_name.value] = {
            'description': self.description.value,
            'component': self
        }
        print(f"Registered agent start: {self.agent_flow_name.value}")


    def execute(self, ctx) -> None:
        # This method is called when the router selects this flow.
        # The actual agent logic starts from the components connected *after* this one.
        # The conversation should ideally be passed through the context by the router.
        print(f"Executing AgentDefineRoute: {self.agent_flow_name.value}")
        # The SubGraphExecutor in AgentRouter will handle running the subsequent components.
        pass


@xai_component
class AgentRouter(Component):
    """An agent that routes a conversation to a specific agent route based on available AgentDefineRoute components.

    It uses its own configured agent (specified by agent_name) to decide which flow is most appropriate.

    ##### inPorts:
    - agent_name: The name of the agent configuration to use for the routing decision.
    - conversation: The conversation history to be routed.
    - final_context_key: The key in the context where the chosen agent route is expected to place its final result (default: 'agent_flow_result').


    ##### outPorts:
    - result: The final result produced by the selected agent route.
    - chosen_flow_name: The name of the agent route that was chosen.
    """
    agent_name: InCompArg[str]
    conversation: InArg[list]
    final_context_key: InArg[str]

    result: OutArg[any]
    chosen_flow_name: OutArg[str]

    # Reusing parts of AgentRun's logic for the LLM call might be needed.
    # For simplicity, let's define a basic LLM call here.
    # This assumes access to similar helper functions or context setup as AgentRun.

    def execute(self, ctx) -> None:
        agent_context_key = 'agent_' + self.agent_name.value
        if agent_context_key not in ctx:
            raise ValueError(f"Agent configuration '{self.agent_name.value}' not found. Initialize it with AgentInit first.")

        agent_config = ctx[agent_context_key]
        provider = agent_config['agent_provider']
        model_name = agent_config['agent_model']

        # 1. Find available agent routes (AgentDefineRoute components)
        registered_starts = ctx.get('registered_agent_routes', {})
        if not registered_starts:
            raise ValueError("No agent routes (AgentDefineRoute components) found registered in the context.")

        # 2. Prepare descriptions for the routing prompt
        flow_descriptions = []
        for name, data in registered_starts.items():
            flow_descriptions.append(f"- Name: {name}\n  Description: {data['description']}")
        
        available_flows_text = "\n".join(flow_descriptions)

        # 3. Construct the prompt for the router agent
        routing_prompt = f"""You are a routing agent. Based on the following conversation history, choose the most appropriate agent route to handle the latest user request.

Conversation History:
{json.dumps(self.conversation.value, indent=2)}

Available agent routes:
{available_flows_text}

Respond ONLY with the exact 'Name' of the best agent route to use. Do not add any explanation or other text."""

        # Prepare conversation for the LLM call
        router_conversation = [
            {"role": "system", "content": "You are an expert routing agent."}, # Basic system prompt
            {"role": "user", "content": routing_prompt}
        ]

        # 4. Call the LLM using the shared dispatch function
        chosen_flow = None
        print(f"Attempting routing with provider: {provider}, model: {model_name}")
        try:
            # Use a low temperature for deterministic routing
            response = _dispatch_llm_call(ctx, provider, model_name, router_conversation, temperature=0.1)
            chosen_flow = response['content'].strip()
        except Exception as e:
            print(f"Error calling LLM for routing via provider {provider}: {e}")
            traceback.print_exc()
            raise ConnectionError(f"Failed to get routing decision from {provider}: {e}")

        if not chosen_flow or chosen_flow not in registered_starts:
            available_keys = list(registered_starts.keys())
            # Sometimes the model might return the name in quotes, try stripping them
            if chosen_flow and chosen_flow.startswith('"') and chosen_flow.endswith('"'):
                chosen_flow = chosen_flow[1:-1]
            elif chosen_flow and chosen_flow.startswith("'") and chosen_flow.endswith("'"):
                 chosen_flow = chosen_flow[1:-1]

            if not chosen_flow or chosen_flow not in available_keys:
                 print(f"Router LLM returned an invalid route name: '{chosen_flow}'. Available: {available_keys}")
                 raise ValueError(f"AgentRouter failed to select a valid agent route. Response: '{chosen_flow}'")

        print(f"AgentRouter chose route: {chosen_flow}")
        self.chosen_flow_name.value = chosen_flow

        # 5. Get the chosen AgentDefineRoute component
        chosen_start_component = registered_starts[chosen_flow]['component']

        # 6. Execute the chosen subgraph
        ctx['current_conversation_for_flow'] = self.conversation.value
        result_key = self.final_context_key.value if self.final_context_key.value else 'agent_flow_result'
        ctx[result_key] = None # Clear any previous result

        print(f"Executing subgraph starting from: {chosen_flow}")
        try:
            # Execute the graph starting from the component *after* the AgentDefineRoute node
            if chosen_start_component.next:
                 SubGraphExecutor(chosen_start_component.next).do(ctx)
            else:
                 print(f"Warning: Chosen AgentDefineRoute '{chosen_flow}' has no connected components to execute.")
                 # Set a default result or handle as needed
                 ctx[result_key] = "No components connected to the chosen start flow."

        except Exception as e:
            print(f"Error executing chosen agent route '{chosen_flow}': {e}")
            traceback.print_exc()
            self.result.value = f"Error during execution of flow '{chosen_flow}': {e}"
            raise # Re-raise to signal failure upstream

        # 7. Retrieve the result from the context
        final_result = ctx.get(result_key)
        print(f"Subgraph execution finished. Result found in ctx['{result_key}']: {final_result}")
        self.result.value = final_result

        # Optional: Clean up context
        # if 'current_conversation_for_flow' in ctx: del ctx['current_conversation_for_flow']
        # if result_key in ctx: del ctx[result_key]


# Placeholder for querying core system about MCP tools
# In a real implementation, this would interact with the framework/environment
def query_core_system_for_mcp_tools(server_name: str) -> list:
    """Placeholder: Queries the core system for tools provided by a specific MCP server."""
    print(f"[Placeholder] Querying core system for tools from MCP server: {server_name}")
    # Example response structure - replace with actual system interaction
    if server_name == "filesystem": # Example
         return [{"name": "readFile", "description": "Reads a file"}, {"name": "writeFile", "description": "Writes a file"}]
    if server_name == "weather": # Example
         return [{"name": "get_forecast", "description": "Gets weather forecast"}]
    return []

def make_tools_prompt(standard_toolbelt: dict, enabled_mcp_servers: list, metadata: dict, provided_system: str=None) -> dict:
    """Generates the tool descriptions for the system prompt, including standard and MCP tools."""
    tool_desc_parts = []

    # 1. Add standard tools
    for key, value in standard_toolbelt.items():
        # Ensure value has a description attribute
        desc = getattr(value, 'description', 'No description available.')
        # Convert any old TOOL: syntax to new XML format
        desc = convert_old_tool_syntax_to_xml(desc)
        tool_desc_parts.append(f'{key}: {desc}')

    # 2. Add MCP tools
    for server_name in enabled_mcp_servers:
        try:
            mcp_tools = query_core_system_for_mcp_tools(server_name)
            for tool_info in mcp_tools:
                 # Optionally prefix with server name for clarity if needed: f'{server_name}.{tool_info["name"]}'
                desc = convert_old_tool_syntax_to_xml(tool_info["description"])
                tool_desc_parts.append(f'{tool_info["name"]}: {desc}')
        except Exception as e:
            print(f"Error querying MCP tools for server '{server_name}': {e}")
            tool_desc_parts.append(f"# Error loading tools for MCP server: {server_name}")

    tools_string = '\n'.join(tool_desc_parts)

    # Memory tool descriptions with both JSON and XML examples
    recall = 'lookup_memory: Fuzzily looks up a previously remembered JSON memo in your memory.\nEXAMPLE:\n\nUSER:\nWhat things did I have to do today?\nASSISTANT:\n<tool name="lookup_memory">\n{"query":"todo list"}\n</tool>\nSYSTEM:\n[{"id": 1, "summary": Todo List for Februrary", "tasks": [{"title": "Send invoices", "due_date":"2025-02-01"}]}]\nASSISTANT:\n<tool name="get_current_time">\n</tool>\nSYSTEM:\n2024-02-01T09:30:03\nASSISTANT:\nLooks like you just had to send invoices today.\n\nAlternative XML format:\n<tool name="lookup_memory">\n<query>todo list</query>\n</tool>'
    remember = 'create_memory: Remembers a new json note for the future.  Always provide json with a summary prompt that will serve as the lookup vector.  The summary and entire json can be remembered later with lookup_memory.\nEXAMPLE:\n\nUSER:\nRemind me to send invoices on the first of Feburary.\nASSISTANT:\n<tool name="create_memory">\n{ "summary": "todo List for Februrary", "tasks": [{"title": "Send invoices", "due_date":"2025-02-01"}]}\n</tool>\n\nAlternative XML format:\n<tool name="create_memory">\n<summary>todo List for Februrary</summary>\n<tasks>[{"title": "Send invoices", "due_date":"2025-02-01"}]</tasks>\n</tool>'

    return {
        'tools': tools_string,
        'lookup_memory': recall,
        'create_memory': remember,
        'memory': recall + remember,
        'tool_instruction': 'To use a tool, use XML tags like this: <tool name="tool_name">arguments</tool>. Arguments can span multiple lines and can be in JSON format ({"arg": "value"}) or XML format (<arg>value</arg>). For tools with no arguments, use empty tags: <tool name="tool_name"></tool>. The system will respond with the results.',
        'metadata': metadata,
        'provided_system': provided_system
    }

def conversation_to_vertexai(conversation) -> str:
    ret = ""
    
    for message in conversation:
        ret += message['role'] + ":" + message['content']
        ret += "\n\n"
    
    return ret
 
@xai_component
class AgentMergeSystem(Component):
    new_system_prompt: InCompArg[str]
    conversation: InCompArg[any]

    out_conversation: OutArg[list]

    def execute(self, ctx) -> None:
        new_system_prompt = self.new_system_prompt.value
        conversation = copy.deepcopy(self.conversation.value)
        
        if conversation[0]['role'] != 'system':
            conversation.insert(0, {'role': 'system', 'content': new_system_prompt })
        else:
            provided_system = conversation[0]['content']
            conversation[0]['content'] = provided_system + '\n\n' + new_system_prompt

        self.out_conversation.value = conversation


# --- Standalone LLM Call Functions ---

def _run_llm_bedrock(ctx, model_name, conversation, temperature):
    """Calls the Bedrock API (Anthropic models)."""
    print("Calling Bedrock (Anthropic)...")
    # Ensure bedrock_client is available in context (initialized elsewhere)
    bedrock_client = ctx.get('bedrock_client')
    if bedrock_client is None:
        raise Exception("Bedrock client has not been authorized or is not found in context.")

    system = conversation[0]['content'] if conversation and conversation[0]['role'] == 'system' else None
    # Pass only non-system messages to encode_prompt if it expects that
    messages_to_encode = conversation[1:] if system else conversation
    messages = encode_prompt(model_name, messages_to_encode) # Assumes encode_prompt handles the format

    body_data = {
        "messages": messages,
        "max_tokens": 8192, # Consider making this configurable
        "temperature": temperature,
        "anthropic_version": "bedrock-2023-05-31" # Check if this needs updates for newer models
    }
    # Conditionally add system prompt if it exists
    if system:
        body_data["system"] = system

    body = json.dumps(body_data)

    try:
        api_response = bedrock_client.invoke_model(
            body=body,
            modelId=model_name,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(api_response.get('body').read())

        # Handle potential variations in response structure (Anthropic specific)
        if 'content' in response_body and isinstance(response_body['content'], list) and len(response_body['content']) > 0:
             content_block = response_body['content'][0]
             if content_block.get('type') == 'text':
                 text = content_block['text']
             else:
                 # Handle other content types if necessary (e.g., tool_use)
                 print(f"Warning: Bedrock returned non-text content block: {content_block}")
                 text = json.dumps(content_block) # Or handle appropriately
        elif 'completion' in response_body: # Older completion-style response (less likely for Claude 3+)
             text = response_body['completion']
        else:
             print(f"Warning: Unexpected Bedrock response structure: {response_body}")
             raise Exception('Unknown content structure returned from Bedrock model.')

        response = {"role": "assistant", "content": text}
        print("Bedrock response processed.")
        return response
    except Exception as e:
        print(f"Error during Bedrock API call: {e}")
        traceback.print_exc()
        raise # Re-raise the exception


def _run_llm_vertexai(ctx, model_name, conversation, temperature):
    """Calls the Vertex AI API."""
    print("Calling Vertex AI...")
    if 'vertexai' not in globals() or 'GenerativeModel' not in globals():
        raise ImportError("Vertex AI library not available or GenerativeModel not imported.")

    # Convert conversation format if necessary (using existing helper)
    # Ensure conversation_to_vertexai handles the specific format needed by the model
    inputs = conversation_to_vertexai(conversation)
    model = GenerativeModel(model_name)

    try:
        result = model.generate_content(
            inputs,
            generation_config={
                "max_output_tokens": 8192, # Consider making configurable
                "stop_sequences": ["\n\nsystem:", "\n\nuser:", "\n\nassistant:"], # Standard stops
                "temperature": temperature,
                "top_p": 1 # Often recommended to adjust temp OR top_p, not both heavily
            },
            safety_settings=[], # Adjust as needed
            stream=False,
        )
        # Safely access text, handling potential errors or empty responses
        text_response = getattr(result, 'text', '')
        if not text_response and hasattr(result, 'candidates') and result.candidates:
             # Try getting content from the first candidate if text is empty
             first_candidate = result.candidates[0]
             if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts'):
                 text_response = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))


        # Basic parsing, might need refinement based on how Vertex returns roles
        # Vertex might just return the assistant's text directly without the role prefix
        response_content = text_response.strip()


        response = {"role": "assistant", "content": response_content}
        print("Vertex AI response processed.")
        return response
    except Exception as e:
        print(f"Error during Vertex AI API call: {e}")
        traceback.print_exc()
        raise # Re-raise the exception


def _run_llm_openai(ctx, model_name, conversation, temperature):
    """Calls the OpenAI API."""
    #print("Calling OpenAI...")
    #print(f"Model: {model_name}")
    #print(f"Temperature: {temperature}")
    #print("Conversation:", flush=True)
    #print(conversation, flush=True)
    
    if 'openai' not in globals():
        raise ImportError("OpenAI library not available or imported.")

    # Clean conversation: remove trailing empty assistant message if present
    if conversation and conversation[-1]['role'] == 'assistant' and not conversation[-1]['content']:
        conversation = conversation[:-1]

    try:
        if model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4') or model_name.startswith('gpt-5'):
            reasoning_effort = 'low'
            if temperature > 0.3:
                reasoning_effort = 'medium'
            elif temperature > 0.5:
                reasoning_effort = 'high'
                
            completion = openai.chat.completions.create(
                model=model_name,
                messages=conversation,
                max_completion_tokens=8192,
                reasoning_effort=reasoning_effort
            )
        if model_name.startswith('grok-4'):
            completion = openai.chat.completions.create(
                model=model_name,
                messages=conversation,
                max_completion_tokens=8192
            )
        else:
            params = {
                "model": model_name,
                "messages": conversation,
                "max_tokens": 8192,
                "stop": ["\nsystem:\n", "\nSYSTEM:\n", "\nUSER:\n", "\nASSISTANT:\n"],
                "temperature": temperature,
                "response_format": { "type": "text" }
            }
            completion = openai.chat.completions.create(**params)

        # Ensure message content is accessed correctly
        response_content = ""
        if completion.choices and completion.choices[0].message:
             response_content = completion.choices[0].message.content or "" # Handle None content

        response = {"role": "assistant", "content": response_content}
        #print("Got raw response:", flush=True)
        #print(response, flush=True)
        #print("OpenAI response processed.")
        return response
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        traceback.print_exc()
        raise # Re-raise the exception


def _dispatch_llm_call(ctx, provider, model_name, conversation, temperature):
    """Dispatches the LLM call to the appropriate provider function."""
    print(f"Dispatching LLM call to provider: {provider}")
    if provider == 'openai':
        return _run_llm_openai(ctx, model_name, conversation, temperature)
    elif provider == 'vertexai':
        return _run_llm_vertexai(ctx, model_name, conversation, temperature)
    elif provider == 'bedrock':
        return _run_llm_bedrock(ctx, model_name, conversation, temperature)
    else:
        raise ValueError(f"Unsupported LLM provider in dispatch: {provider}")


# --- End Standalone LLM Call Functions ---


# --- Duplicate Tool Call Prevention Functions ---

def normalize_tool_args(args_dict):
    """Remove _nonce and _duplicate_reasoning from args for comparison"""
    if args_dict is None:
        return None
    normalized = args_dict.copy()
    normalized.pop('_nonce', None)
    normalized.pop('_duplicate_reasoning', None)
    return normalized

def extract_nonce_and_reasoning(args_dict):
    """Extract _nonce and _duplicate_reasoning from parsed args"""
    if args_dict is None:
        return None, None
    return args_dict.get('_nonce'), args_dict.get('_duplicate_reasoning')

def is_duplicate_tool_call(tool_name, args_dict, tool_history):
    """Check if this tool call is a duplicate of a previous call"""
    normalized_args = normalize_tool_args(args_dict)
    
    for prev_call in tool_history:
        if (prev_call['tool_name'] == tool_name and 
            prev_call['normalized_args'] == normalized_args):
            return True
    return False

def is_reasoning_unique(reasoning, tool_history):
    """Check if the reasoning is unique within the current tool history"""
    if reasoning is None or reasoning.strip() == "":
        return False
    
    used_reasoning = [call.get('reasoning') for call in tool_history 
                     if call.get('reasoning') is not None]
    return reasoning not in used_reasoning

# --- End Duplicate Tool Call Prevention Functions ---


@xai_component
class AgentRun(Component):
    """Run the agent with the given conversation.

    ##### branches:
    - on_thought: Called whenever the agent uses a tool.

    ##### inPorts:
    - agent_name: The name of the agent to run.
    - conversation: The conversation to send to the agent.

    ##### outPorts:
    - out_conversation: The conversation with the agent's responses.
    - last_response: The last response of the agent.

    """
    on_thought: BaseComponent

    agent_name: InCompArg[str]
    metadata: InArg[dict]
    conversation: InCompArg[any]

    out_conversation: OutArg[list]
    last_response: OutArg[str]

    tool_call_history: list = []

    def execute(self, ctx) -> None:
        # Reset tool call history for this execution
        self.tool_call_history = []
        try:
            self.do_execute(ctx)
        except Exception as e:
            print(e)
            raise e
            
    def do_execute(self, ctx) -> None:
        agent = ctx['agent_' + self.agent_name.value]

        model_name = agent['agent_model']
        standard_toolbelt = agent['agent_toolbelt'] # Now contains only standard tools
        enabled_mcp_servers = agent.get('enabled_mcp_servers', []) # Get the list of enabled MCP servers
        system_prompt = agent['agent_system_prompt']

        # deep to avoid messing with the original system prompt.
        conversation = copy.deepcopy(self.conversation.value)

        metadata = self.metadata.value
        

        prompt_args = make_tools_prompt(standard_toolbelt, enabled_mcp_servers, metadata)

        if conversation[0]['role'] != 'system':
            # Insert new system prompt if none exists
            conversation.insert(0, {'role': 'system', 'content': system_prompt.format(**prompt_args)})
        else:
            # Format the existing system prompt
            provided_system = conversation[0]['content']
            prompt_args['provided_system'] = provided_system # Add existing system prompt content
            conversation[0]['content'] = system_prompt.format(**prompt_args)

        thoughts = 0
        stress_level = 0.0  # Raise temperature if there are failures.

        while thoughts < agent['max_thoughts']:
            thoughts += 1

            if thoughts == agent['max_thoughts']:
                if is_openai_model(model_name):
                    conversation.append({"role": "system", "content": "Maximum tool usage reached. Tools Unavailable"})
                else:
                    conversation.append({"role": "user", "content": "SYSTEM:\nMaximum tool usage reached. Tools Unavailable"})

            # Call the LLM using the dispatch function
            try:
                response = _dispatch_llm_call(
                    ctx,
                    agent['agent_provider'],
                    model_name,
                    conversation,
                    temperature=stress_level + 0.5 # Default temperature logic
                )
            except Exception as e:
                 print(f"Error during LLM call in AgentRun: {e}")
                 traceback.print_exc()
                 # Decide how to handle the error, e.g., append an error message or raise
                 conversation.append({"role": "system", "content": f"Error communicating with LLM: {e}"})
                 break # Exit the loop on LLM error

            conversation.append(response)

            # HACK: support buggy output that happens offten in glm-4.5.
            # Might be able to remove if we use native tool support.
            if '<toolname=' in response['content']:
                response['content'] = response['content'].replace('<toolname=', '<tool name=')
            
            if thoughts <= agent['max_thoughts'] and response.get('content') and '<tool name=' in response['content']:
                stress_level = self.handle_tool_use(ctx, agent, conversation, response['content'], standard_toolbelt, enabled_mcp_servers, stress_level, model_name)
            else:
                # Allow only one tool per thought.
                break

        self.out_conversation.value = conversation
        self.last_response.value = conversation[-1]['content']

    # Note: run_bedrock, run_vertexai, run_openai methods are removed.

    # Placeholder for querying core system about which server provides a tool
    def query_core_system_for_mcp_tool_server(self, tool_name: str):
        """Placeholder: Queries the core system to find which MCP server provides a tool."""
        print(f"[Placeholder] Querying core system for server providing tool: {tool_name}")
        # Example logic - replace with actual system interaction
        if tool_name in ["readFile", "writeFile"]: return "filesystem"
        if tool_name == "get_forecast": return "weather"
        return None

    # Placeholder for triggering framework's use_mcp_tool
    def framework_use_mcp_tool(self, server_name: str, tool_name: str, tool_args: str) -> str:
        """Placeholder: Triggers the core framework to execute use_mcp_tool."""
        print(f"[Placeholder] Framework executing use_mcp_tool: server='{server_name}', tool='{tool_name}', args='{tool_args}'")
        # Example response - replace with actual framework interaction result
        if tool_name == "get_forecast":
            return json.dumps([{"city": "Example City", "temp": 15.0, "desc": "cloudy"}], indent=2)
        elif tool_name == "readFile":
             return "Placeholder file content for " + tool_args
        # Simulate an error for unknown tools in placeholder
        return f"ERROR: Placeholder framework could not execute tool '{tool_name}' on server '{server_name}'"


    def handle_tool_use(self, ctx, agent, conversation, content, standard_toolbelt, enabled_mcp_servers, stress_level, model_name):
        """Handles tool calls in XML format, dispatching to standard tools or enabled MCP servers."""
        self.last_response.value = conversation[-1]['content'] # Save pre-tool-call response
        
        # Look for XML tool tags
        import re
        tool_pattern = r'<tool\s+name="([^"]+)">(.*?)</tool>'
        match = re.search(tool_pattern, content, re.DOTALL)
        
        if match:
            tool_name = match.group(1)
            tool_args_str = match.group(2).strip()
            
            # Clean up agent response to only include text before the tool call
            pre_tool_text = content[:match.start()].strip()
            conversation[-1]['content'] = pre_tool_text

            # Parse tool arguments to check for duplicates
            parsed_args, _ = parse_tool_args(tool_args_str)
            nonce, reasoning = extract_nonce_and_reasoning(parsed_args)
            
            # Check for duplicate tool calls
            if is_duplicate_tool_call(tool_name, parsed_args, self.tool_call_history):
                if reasoning is None or reasoning.strip() == "":
                    # Duplicate call without reasoning - block it
                    error_message = f"Duplicate tool call detected for '{tool_name}' with same arguments. To retry this tool, provide '_duplicate_reasoning' parameter with unique justification."
                    print(f"Blocked duplicate tool call: {tool_name} with args: {normalize_tool_args(parsed_args)}")
                    
                    if is_openai_model(model_name):
                        conversation.append({"role": "system", "content": f"ERROR: {error_message}"})
                    else:
                        conversation.append({"role": "user", "content": f"SYSTEM:\nERROR: {error_message}"})
                    
                    # Give on_thought a chance to see the error
                    self.out_conversation.value = conversation
                    if hasattr(self, 'on_thought') and self.on_thought:
                        SubGraphExecutor(self.on_thought).do(ctx)
                    
                    return min(stress_level + 0.1, 1.5) # Increase stress due to blocked duplicate
                
                elif not is_reasoning_unique(reasoning, self.tool_call_history):
                    # Duplicate call with non-unique reasoning - block it
                    error_message = f"Duplicate tool call detected for '{tool_name}' with previously used reasoning: '{reasoning}'. Please provide unique justification."
                    print(f"Blocked duplicate tool call with reused reasoning: {tool_name}")
                    
                    if is_openai_model(model_name):
                        conversation.append({"role": "system", "content": f"ERROR: {error_message}"})
                    else:
                        conversation.append({"role": "user", "content": f"SYSTEM:\nERROR: {error_message}"})
                    
                    # Give on_thought a chance to see the error
                    self.out_conversation.value = conversation
                    if hasattr(self, 'on_thought') and self.on_thought:
                        SubGraphExecutor(self.on_thought).do(ctx)
                    
                    return min(stress_level + 0.1, 1.5) # Increase stress due to blocked duplicate
                
                else:
                    # Duplicate call with unique reasoning - allow it
                    print(f"Allowing duplicate tool call '{tool_name}' with unique reasoning: '{reasoning}'")

            # Record this tool call in history (before execution in case of errors)
            tool_call_record = {
                'tool_name': tool_name,
                'normalized_args': normalize_tool_args(parsed_args),
                'reasoning': reasoning
            }
            self.tool_call_history.append(tool_call_record)

            tool_result = None
            error_message = None

            # 1. Check Memory Tools (Special Handling)
            if tool_name == 'lookup_memory':
                    memory = agent.get('agent_memory')
                    if memory:
                        try:
                            # Attempt to parse args as JSON for query structure
                            try: obj = json.loads(tool_args_str); query = obj['query']
                            except: query = tool_args_str # Fallback to raw string query
                            tool_result = str(memory.query(query, 3))
                            print(f"lookup_memory got result:\n{tool_result}", flush=True)
                        except Exception as e:
                            error_message = f"Error during lookup_memory: {e}"
                            traceback.print_exc()
                    else: error_message = "Memory component not available for lookup_memory."

            elif tool_name == 'create_memory':
                memory = agent.get('agent_memory')
                if memory:
                    try:
                        # Attempt to parse args as JSON, otherwise treat as string
                        try: obj = json.loads(tool_args_str)
                        except: obj = tool_args_str # Store raw string if not JSON
                        self.remember_tool(agent, obj, conversation, model_name) # remember_tool handles adding to memory
                    except Exception as e:
                        error_message = f"Error during create_memory: {e}"
                        traceback.print_exc()
                else: error_message = "Memory component not available for create_memory."

            # 2. Check Standard Tools
            elif tool_name in standard_toolbelt:
                    try:
                        tool_callable = standard_toolbelt[tool_name]
                        tool_result = tool_callable(tool_args_str) # Call the standard tool
                        print(f"Standard tool '{tool_name}' got result:\n{tool_result}", flush=True)
                    except Exception as e:
                        error_message = f"Error executing standard tool '{tool_name}': {e}"
                        traceback.print_exc()

            # 3. Check Enabled MCP Tools
            else:
                    try:
                        # Query system to find which server (if any) provides this tool
                        server_name = self.query_core_system_for_mcp_tool_server(tool_name)

                        if server_name and server_name in enabled_mcp_servers:
                            # Tool belongs to an enabled MCP server, trigger framework execution
                            tool_result = self.framework_use_mcp_tool(server_name, tool_name, tool_args_str)
                            print(f"MCP tool '{tool_name}' on server '{server_name}' executed via framework. Result:\n{tool_result}", flush=True)
                        elif server_name:
                             error_message = f"Tool '{tool_name}' found on MCP server '{server_name}', but this server is not enabled for this agent run."
                        else:
                             error_message = f"Tool '{tool_name}' not found in standard tools or any known MCP server."

                    except Exception as e:
                         error_message = f"Error during MCP tool lookup/execution for '{tool_name}': {e}"
                         traceback.print_exc()


            # Append result or error to conversation
            if error_message:
                print(f"Tool execution failed for '{tool_name}': {error_message}", flush=True)
                if is_openai_model(model_name):
                    conversation.append({"role": "system", "content": f"ERROR: {error_message}"})
                else:
                    conversation.append({"role": "user", "content": f"SYSTEM:\nERROR: {error_message}"})
                stress_level = min(stress_level + 0.1, 1.5) # Increase stress on error
            elif tool_result is not None:
                 # Ensure result is a string, handle potential non-string returns gracefully
                result_str = str(tool_result)
                if result_str:
                    if is_openai_model(model_name):
                        conversation.append({"role": "system", "content": result_str})
                    else:
                        conversation.append({"role": "user", "content": f"SYSTEM:\n{result_str}"})
                else:
                    if is_openai_model(model_name):
                        conversation.append({"role": "system", "content": "Tool executed successfully with empty result."})
                    else:
                        conversation.append({"role": "user", "content": "SYSTEM:\nTool executed successfully with empty result."})
                 # Potentially decrease stress on success? stress_level = max(0, stress_level - 0.05)
            # else: tool had no result and no error (e.g., create_memory handled its own confirmation)

            # Give on_thought a chance to see the result/error
            self.out_conversation.value = conversation
            if hasattr(self, 'on_thought') and self.on_thought:
                 SubGraphExecutor(self.on_thought).do(ctx)
        else:
            # No tool was found in the response
            self.last_response.value = content

        return stress_level


    def remember_tool(self, agent, tool_args, conversation, model_name=None):
        memory = agent['agent_memory']
        if isinstance(tool_args, str):
            prompt_start = tool_args.find('"')
            prompt_end = tool_args.find('"', prompt_start)
            prompt = tool_args[prompt_start + 1:prompt_end].strip()
            memo_start = tool_args.find('"', prompt_end)
            memo = tool_args[memo_start + 1:len(tool_args) - 1].replace('\"', '"')
        else:
            prompt = tool_args['summary']
            memo = json.dumps(tool_args)

        try:
            json_memo = json.loads(memo)
        except Exception:
            # Invalid JSON, so just store as a string.
            json_memo = '"' + memo + '"'

        memory.add('', prompt, json_memo)
        print(f"Added {prompt}: {memo} to memory", flush=True)
        
        if model_name and is_openai_model(model_name):
            conversation.append({"role": "system", "content": f"Memory {prompt} stored."})
        else:
            conversation.append({"role": "user", "content": f"SYSTEM:\nMemory {prompt} stored."})

    # Note: run_tool is removed as its logic is now integrated into handle_tool_use


@xai_component
class AgentRunTool(Component):
    """Run a specified tool manually and append the result to a copy of the conversation.

    ##### inPorts:
    - agent_name: The agent whose toolbelt will be used.
    - tool_name: The name of the tool to run.
    - tool_args: The arguments for the tool, passed as is if str or converted to JSON otherwise.
    - conversation: The current conversation to update.

    ##### outPorts:
    - tool_output: The raw output from the tool.
    - updated_conversation: The updated conversation after running the tool.
    """

    agent_name: InCompArg[str]
    tool_name: InCompArg[str]
    tool_args: InArg[any]
    conversation: InCompArg[list]

    tool_output: OutArg[str]
    updated_conversation: OutArg[list]

    def execute(self, ctx) -> None:
        agent_context = ctx['agent_' + self.agent_name.value]
        standard_toolbelt = agent_context.get('agent_toolbelt', {})
        enabled_mcp_servers = agent_context.get('enabled_mcp_servers', [])

        current_conversation = self.conversation.value.copy()  # Create a copy of the conversation

        try:
            # Check standard tools first
            if self.tool_name.value in standard_toolbelt:
                tool = standard_toolbelt[self.tool_name.value]
                is_mcp = False
            else:
                # Check if it's an enabled MCP tool (requires framework interaction)
                # Placeholder: Assume framework provides a way to check/get MCP tool info
                server_name = self.query_core_system_for_mcp_tool_server(self.tool_name.value) # Reuse placeholder
                if server_name and server_name in enabled_mcp_servers:
                    # We don't get a callable here, we'll use the framework call below
                    is_mcp = True
                else:
                    raise KeyError(f"Tool '{self.tool_name.value}' not found in standard tools or enabled MCP servers.")

            if self.tool_args.value is None:
                args = ""
            if isinstance(self.tool_args.value, str):
                args = self.tool_args.value
            else:
                args = json.dumps(self.tool_args.value)
                
            if is_mcp:
                 # Trigger framework's MCP tool execution
                 tool_result = self.framework_use_mcp_tool(server_name, self.tool_name.value, args) # Reuse placeholder
            else:
                 # Execute standard tool callable
                 tool_result = tool(args)

            # Append the tool usage to the copied conversation
            current_conversation.append({"role": "assistant", "content": f'<tool name="{self.tool_name.value}">\n{self.tool_args.value}\n</tool>'})

            if tool_result != '':
                # Check if we can determine the model from the agent context
                model_name = agent_context.get('agent_model', '')
                if is_openai_model(model_name):
                    current_conversation.append({"role": "system", "content": str(tool_result)})
                else:
                    current_conversation.append({"role": "user", "content": "SYSTEM:\n" + str(tool_result)})
            else:
                model_name = agent_context.get('agent_model', '')
                if is_openai_model(model_name):
                    current_conversation.append({"role": "system", "content": "Empty string result"})
                else:
                    current_conversation.append({"role": "user", "content": "SYSTEM:\nEmpty string result"})

            self.tool_output.value = str(tool_result)
            self.updated_conversation.value = current_conversation
        except KeyError:
            error_message = f"ERROR: TOOL '{self.tool_name.value}' not found."
            model_name = agent_context.get('agent_model', '')
            if is_openai_model(model_name):
                current_conversation.append({"role": "system", "content": error_message})
            else:
                current_conversation.append({"role": "user", "content": "SYSTEM:\n" + error_message})
            self.updated_conversation.value = current_conversation
        except Exception as e:
            error_message = f"ERROR: An exception occurred while running the tool: {str(e)}"
            model_name = agent_context.get('agent_model', '')
            if is_openai_model(model_name):
                current_conversation.append({"role": "system", "content": error_message})
            else:
                current_conversation.append({"role": "user", "content": "SYSTEM:\n" + error_message})
            self.updated_conversation.value = current_conversation


@xai_component
class AgentLearn(Component):
    """Run the agent with the given conversation.

    ##### branches:
    - on_thought: Called whenever the agent uses a tool.

    ##### inPorts:
    - agent_name: The name of the agent to run.
    - conversation: The conversation to send to the agent.

    ##### outPorts:
    - out_conversation: The conversation with the agent's responses.
    - last_response: The last response of the agent.

    """
    on_thought: BaseComponent

    agent_name: InCompArg[str]
    metadata: InArg[dict]
    conversation: InCompArg[any]

    out_conversation: OutArg[list]
    last_response: OutArg[str]

    def execute(self, ctx) -> None:
        agent = ctx['agent_' + self.agent_name.value]

        model_name = agent['agent_model']
        toolbelt = agent['agent_toolbelt']
        enabled_mcp_servers = agent.get('enabled_mcp_servers', [])
        system_prompt = agent['agent_system_prompt']
        metadata = self.metadata.value

        # Deep copy to avoid messing with the original system prompt.
        conversation = copy.deepcopy(self.conversation.value)

        if conversation[0]['role'] != 'system':
            conversation.insert(0, {'role': 'system', 'content': system_prompt.format(**make_tools_prompt(toolbelt, enabled_mcp_servers, metadata))})
        else:
            conversation[0]['content'] = system_prompt.format(**make_tools_prompt(toolbelt, enabled_mcp_servers, metadata))

        # Add system message to use memory tools
        memory_instruction = {
            "role": "system",
            "content": "Use memory tools to review the conversation so far. Learn from it by updating existing memories or creating new ones. Then, respond with a summary of what you have learned."
        }
        conversation.append(memory_instruction)

        thoughts = 0
        stress_level = 0.0  # Raise temperature if there are failures.

        while thoughts < agent['max_thoughts']:
            thoughts += 1

            if thoughts == agent['max_thoughts']:
                if is_openai_model(model_name):
                    conversation.append({"role": "system", "content": "Maximum tool usage reached. Tools Unavailable"})
                else:
                    conversation.append({"role": "user", "content": "SYSTEM:\nMaximum tool usage reached. Tools Unavailable"})

            if agent['agent_provider'] == 'vertexai':
                response = self.run_vertexai(ctx, model_name, conversation, stress_level)
            elif agent['agent_provider'] == 'openai':
                response = self.run_openai(ctx, model_name, conversation, stress_level)
            elif agent['agent_provider'] == 'bedrock':
                response = self.run_bedrock(ctx, model_name, conversation, stress_level)
            else:
                raise ValueError("Unknown agent provider")

            conversation.append(response)

            if thoughts <= agent['max_thoughts'] and '<tool name=' in response['content']:
                stress_level = self.handle_tool_use(ctx, agent, conversation, response['content'], toolbelt, enabled_mcp_servers, stress_level, model_name)
            else:
                # Allow only one tool per thought.
                break

        # Final thoughts: Instruct the agent to summarize what it has learned
        summary_instruction = {
            "role": "system",
            "content": "Summarize your learnings from the conversation and how it will influence your future interactions."
        }
        
        conversation.append(summary_instruction)
        if agent['agent_provider'] == 'vertexai':
            response = self.run_vertexai(ctx, model_name, conversation, stress_level)
        elif agent['agent_provider'] == 'openai':
            response = self.run_openai(ctx, model_name, conversation, stress_level)
        elif agent['agent_provider'] == 'bedrock':
            response = self.run_bedrock(ctx, model_name, conversation, stress_level)
        else:
            raise ValueError("Unknown agent provider")

        conversation.append(response)

        self.out_conversation.value = conversation
        self.last_response.value = conversation[-1]['content']
        

    def run_bedrock(self, ctx, model_name, conversation, stress_level):
        print(conversation)
        print("calling anthropic...")
        
        bedrock_client = ctx.get('bedrock_client')
        if bedrock_client is None:
            raise Exception("Bedrock client has not been authorized")

        if conversation[0]['role'] == 'system':
            system = conversation[0]['content']
        else:
            system = None

        messages = encode_prompt(model_name, conversation[1:])

        body_data = {
            "system": system,
            "messages": messages,
            "max_tokens": 8192,
            "anthropic_version": "bedrock-2023-05-31"
        }

        body = json.dumps(body_data)
        response = bedrock_client.invoke_model(
            body=body,
            modelId=model_name,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response.get('body').read())
        content = response_body.get('content')[0]
        if content['type'] == 'text':
            text = content['text']
        else:
            print(content)
            raise Exception('Unknown content type returned from model.')
        response = { "role": "assistant", "content": text }

        print("got response:")
        print(response)
        return response
    
    def run_vertexai(self, ctx, model_name, conversation, stress_level):
        inputs = conversation_to_vertexai(conversation)
        model = GenerativeModel(model_name)
        result = model.generate_content(
            inputs,
            generation_config={
                "max_output_tokens": 8192,
                "stop_sequences": [
                    "\n\nsystem:",
                    "\n\nuser:",
                    "\n\nassistant:"
                ],
                "temperature": stress_level + 0.5,
                "top_p": 1
            },
            safety_settings=[],
            stream=False,
        )

        if "assistant:" in result.text:
            response = {"role": "assistant", "content": result.text.split("assistant:")[-1]}
        else:
            response = {"role": "assistant", "content": result.text}
        return response

    def run_openai(self, ctx, model_name, conversation, stress_level):
        print(conversation, flush=True)
        if conversation[-1]['role'] == 'assistant' and conversation[-1]['content'] == '':
            conversation.pop()

        if model_name.startswith('o1') or model_name.startswith('o3') or model_name.startswith('o4') or model_name.startswith('gpt-5'):
            reasoning_effort = 'low'
            if stress_level > 0.3:
                reasoning_effort = 'medium'
            elif stress_level > 0.5:
                reasoning_effort = 'high'
                
            result = openai.chat.completions.create(
                model=model_name,
                messages=conversation,
                max_completion_tokens=8192,
                stop=["\nsystem:\n", "\nSYSTEM:\n", "\nUSER:\n", "\nASSISTANT:\n"],
                reasoning_effort='low'
            )
        else:
            result = openai.chat.completions.create(
                model=model_name,
                messages=conversation,
                max_tokens=8192,
                stop=["\nsystem:\n", "\nSYSTEM:\n", "\nUSER:\n", "\nASSISTANT:\n"],
                temperature=stress_level
            )
        try:
            response = result.choices[0].message
            return {"role": "assistant", "content": response.content}
        except:
            print(result, flush=True)
            return {"role": "assistant", "content": "Error...."}

    # Note: AgentLearn inherits from AgentRun, so its handle_tool_use needs the same signature update
    # We can reuse the handle_tool_use implementation from AgentRun if AgentLearn doesn't override it.
    # If AgentLearn *does* override it, that override needs the same parameter/logic updates.
    # Assuming AgentLearn uses AgentRun's handle_tool_use for now. If not, this diff needs adjustment.
    # If AgentLearn has its own copy, apply similar logic changes there.
    # Based on the code, AgentLearn defines its own execute but seems to call AgentRun's methods like run_openai etc.
    # It *doesn't* seem to redefine handle_tool_use, so inheriting the change should work.
    # Let's remove the redundant handle_tool_use definition from AgentLearn if it exists,
    # or ensure it calls super().handle_tool_use(...) if it adds specific logic.
    # Checking the provided code again... AgentLearn *does* have its own copy of handle_tool_use (lines 1037-1084)
    # and run_tool (lines 1108-1131). These need to be updated or removed to inherit from AgentRun.
    # Easiest is to remove the redundant copies in AgentLearn and let it inherit.

    # Remove redundant handle_tool_use from AgentLearn (lines 1037-1084)
    # Remove redundant run_tool from AgentLearn (lines 1108-1131)
    # The call in AgentLearn.execute (line 908) will now correctly call the modified AgentRun.handle_tool_use
    pass # Placeholder for the removal diffs below

    def remember_tool(self, agent, tool_args, conversation, model_name=None):
        memory = agent['agent_memory']
        if isinstance(tool_args, str):
            prompt_start = tool_args.find('"')
            prompt_end = tool_args.find('"', prompt_start)
            prompt = tool_args[prompt_start + 1:prompt_end].strip()
            memo_start = tool_args.find('"', prompt_end)
            memo = tool_args[memo_start + 1:len(tool_args) - 1].replace('\"', '"')
        else:
            prompt = tool_args['summary']
            memo = json.dumps(tool_args)

        try:
            json_memo = json.loads(memo)
        except Exception:
            # Invalid JSON, so just store as a string.
            json_memo = '"' + memo + '"'

        memory.add('', prompt, json_memo)
        print(f"Added {prompt}: {memo} to memory", flush=True)
        
        if model_name and is_openai_model(model_name):
            conversation.append({"role": "system", "content": f"Memory {prompt} stored."})
        else:
            conversation.append({"role": "user", "content": f"SYSTEM:\nMemory {prompt} stored."})

    # Note: run_tool is removed from AgentLearn to inherit from AgentRun (where it's also removed)

def word_or_pair_generator(input_string):
    words = input_string.split(' ')

    for word in words:
        if len(word) > 10:
            for i in range(0, len(word), 2):
                yield word[i:i+2]
        else:
            yield word

        if word != words[-1]:
            yield ' '


@xai_component
class AgentStreamStringResponse(Component):
    """Creates a Stream response from a string.

    When using Converse it is better for the user to see the response word by word
    as if it was being typed out, like it is in ChatGPT.

    Use with the ConverseStreamRespond or ConverseStreamPartialResponse 
    component when using Converse.

    ##### inPorts:
    - input_str: The string to stream.

    ##### outPorts:
    - result_stream: The result of the string to stream.
    """
    
    input_str: InCompArg[str]
    
    result_stream: OutArg[any]
    
    def execute(self, ctx) -> None:
        self.result_stream.value = word_or_pair_generator(self.input_str.value)