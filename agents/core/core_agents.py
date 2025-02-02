# Core agents for system

# Tool-Using Agent
class ToolUseAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, tool_schema: List, tool_agent_type: str) -> None:
        super().__init__("An agent with tools")
        self._system_messages: List[LLMMessage] = [SystemMessage(content=SYSTEM_MESSAGE_CONTENT)]
        self._model_client = model_client
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)

    @message_handler
    async def handle_user_message(self, message: Message, ctx: MessageContext) -> Message:
        """Handles incoming user messages and integrates tool usage."""
        # Build session of messages
        session: List[LLMMessage] = self._system_messages + [UserMessage(content=message.content, source="user")]

        # Call tools if necessary
        messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )

        # Return the final response
        assert isinstance(messages[-1].content, str)
        return Message(content=messages[-1].content)