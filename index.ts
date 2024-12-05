import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import type { RunnableConfig } from "@langchain/core/runnables";
import { ChatOpenAI } from "@langchain/openai";
import { START, END, StateGraph } from "@langchain/langgraph";

const AgentState = Annotation.Root({
  ...MessagesAnnotation.spec,
});

const tools = [new TavilySearchResults({ maxResults: 1 })];

const toolNode = new ToolNode<typeof AgentState.State>(tools);

// Define the LLM to be used in the agent
const llm = new ChatOpenAI({
  temperature: 0,
}).bindTools(tools); // Ensure you bind the same tools passed to the ToolExecutor to the LLM, so these tools can be used in the agent

// Define logic that will be used to determine which conditional edge to go down
const shouldContinue = (data: typeof AgentState.State): "executeTools" | typeof END => {
  const { messages } = data;
  const lastMsg = messages[messages.length - 1];
  // If the agent called a tool, we should continue. If not, we can end.
  if (!("tool_calls" in lastMsg) || !Array.isArray(lastMsg.tool_calls) || !lastMsg?.tool_calls?.length) {
    return END;
  }
  // By returning the name of the next node we want to go to
  // LangGraph will automatically route to that node
  return "executeTools";
};

const callModel = async (data: typeof AgentState.State, config?: RunnableConfig): Promise<Partial<typeof AgentState.State>> => {
  const { messages } = data;
  const result = await llm.invoke(messages, config);
  return {
    messages: [result],
  };
};
// Define a new graph
const workflow = new StateGraph(AgentState)
  // Define the two nodes we will cycle between
  .addNode("callModel", callModel)
  .addNode("executeTools", toolNode)
  // Set the entrypoint as `callModel`
  // This means that this node is the first one called
  .addEdge(START, "callModel")
  // We now add a conditional edge
  .addConditionalEdges(
    // First, we define the start node. We use `callModel`.
    // This means these are the edges taken after the `agent` node is called.
    "callModel",
    // Next, we pass in the function that will determine which node is called next.
    shouldContinue,
  )
  // We now add a normal edge from `tools` to `agent`.
  // This means that after `tools` is called, `agent` node is called next.
  .addEdge("executeTools", "callModel");

const graph = workflow.compile();