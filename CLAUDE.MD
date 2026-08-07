\# Peninsula Ireland Lead CRM Project



\## Tech Stack

\- Frontend: Streamlit

\- Backend: Python 3.10+, Anthropic Python SDK (`anthropic`)

\- Database: SQLite (`sqlite3`)



\## Code Style \& Rules

\- Keep `app.py` modular with clean functional separation (DB handler, API client, UI components).

\- Ensure all Anthropic API calls pass the Master System Prompt verbatim in the `system` parameter.

\- Use `claude-3-7-sonnet-latest` or `claude-3-5-sonnet-latest` for API requests.

\- Always include `web\_search` in the tools parameter when calling the API.

\- Do NOT hardcode API keys; load from environment variables (`ANTHROPIC\_API\_KEY`).



