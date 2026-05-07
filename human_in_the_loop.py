from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt, Command



class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()

@tool
def get_stock_price(symbol:str) -> float:
    ''' 
    Return the current price of a stock given the stock symbol
    :param symbol: stock symbol
    :return: current price of the stock
    '''
    return {
        "MSFT" : 200.3,
        "AAPL" : 100.4,
        "AMZN" : 150.0,
        "RIL" : 87.6
    }.get(symbol,0.0)


@tool
def buy_stocks(symbol: str, quantity: int, total_cost: float):
    '''
    Buy the stocks given the stock symbol and quantity
    '''
    decision = interrupt(f"Approve buying {quantity} shares of {symbol} for a total price of {total_cost}")

    if decision == "yes" :
        return f"You bought {quantity} shares of {symbol} for a total price of {total_cost}"
    else:
        return "Buying declined"


tools = [get_stock_price, buy_stocks]


llm_with_tools = ChatOllama(model="gemma4:e2b").bind_tools(tools)

def chatbot(state:State):
    return { "messages": llm_with_tools.invoke(state['messages']) }

builder = StateGraph(State)

builder.add_node("tools", ToolNode(tools))
builder.add_node("add_chatbot", chatbot)

builder.add_edge(START, "add_chatbot")
builder.add_conditional_edges("add_chatbot", tools_condition)
builder.add_edge("tools", "add_chatbot")

graph = builder.compile(checkpointer=memory)

# from IPython.display import display, Image
# display(Image(graph.get_graph().draw_mermaid_png()))



config1 : RunnableConfig = {
    'configurable' : {
        'thread_id' : '1'
    }
}

msg = "What is total price of 15 RIL stocks?"
state = graph.invoke({ "messages" : [ { "role" : "user", "content": msg } ] }, config=config1)
print(state["messages"][-1].content)


msg = "Buy 15 RIL stocks at current price."
state = graph.invoke({ "messages" : [ { "role" : "user", "content": msg } ] }, config=config1)
print(state.get("__interrupt__"))


decision = input("Approve (yes/no): ")
state = graph.invoke(Command(resume=decision), config=config1)
print(state["messages"][-1].content)


