### Simple chatbot, scripted in Python
# COMPLETED IN UNIVERSITY

### [DEMO VIDEO](https://bit.ly/3pOmbv4)

Created within the module __Human-AI Interaction__, this chatbot implements basic natural language processing (taught as an interactive AI concept). 

### User interaction
The chatbot does not have a seperate UI, existing only in the terminal where the script was run.
1. The user is prompted for their name
2. A loop begins, waiting for a user query. 
3. If the query is STOP (capital sensitive), the loop ends.
4. Otherwise, the chatbot responds as closely related to the query as possible.
5. Repeat to step 2.

### What does it do?
This chatbot uses a spreadsheet of queries and replies to generate a response to anything the user may say. It uses the spreadsheets as they contain possible queries the user could ask, as well as the responses to them. The responses are based on a __similarity index__ between the user-input query compared to all the possible queries. There are two spreadsheets and, essentially, two modes the chatbot has.
1. Question-Answer _(qna_datasheet.csv)_
2. Conversational _(chat_datasheet.csv)_

Even though it's based on a spreadsheet and expectedly limited, the consideration of the time necessary to compare **all** queries could be quite large. To reduce the runtime, we cut the amount of queries necessary to compare by half (roughly).

### In Summary:
1. It __classifies__ whether the user input is a 'question-answer' or 'conversational' query.
2. Based on the classification, it _compares_ the user query to the queries in the relevant spreadsheet. This results in a __similarity index__ for all those queries.
3. The most similar query (closest to 1.0) will be chosen as the __correct choice__ for the chatbot.
4. The answer supplied to the most similar query is then shown to the user. In all cases, there are __multiple responses possible__ for the chatbot to use which will be randomly chosen. This is to simulate a higher-level of knowledge that the chatbot has.
5. The chatbot will ask for a query from the user, and wait.
6. Once recieved, the process will repeat the process.

There are details missing in these steps. For more in-depth analysis of what happens (and why certain methods are chosen), [a report is available here](https://bit.ly/3GInHWx). This report was created during university. 
