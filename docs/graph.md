```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	prepare(prepare)
	agent(agent)
	tools(tools)
	verify(verify)
	answer(answer)
	__end__([<p>__end__</p>]):::last
	__start__ --> prepare;
	agent -.-> answer;
	agent -.-> tools;
	agent -.-> verify;
	prepare --> agent;
	tools -.-> agent;
	tools -.-> answer;
	verify -.-> agent;
	verify -.-> answer;
	answer --> __end__;
	agent -.-> agent;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
