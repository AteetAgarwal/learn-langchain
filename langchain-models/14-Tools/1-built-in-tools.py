from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke('latest news on AI advancements')
print(results)
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

shell_tool = ShellTool()
output = shell_tool.invoke('whoami')
print(output)