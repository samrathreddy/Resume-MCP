# Resume MCP Server

An MCP (Model Context Protocol) server that extracts and formats resume data from PDFs into clean markdown format. (Puch AI Assignment)

## What it does

- Extracts text and hyperlinks from PDF resumes
- Uses OpenAI to format content into structured markdown
- Provides web content fetching capabilities
- Handles authentication and validation

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your credentials:

   ```
   OPENAI_API_KEY=your_openai_api_key
   TOKEN=your_mcp_token
   MY_NUMBER=your_phone_number
   ```

3. **Add your resume**

   Placed PDF resume as `resume.pdf` in the project directory.

## Usage

### Start the server

```bash
python puch_mcp_server.py
```

The server runs on `http://0.0.0.0:8085`

### Test the server

```bash
python test_fastmcp_client.py
```

### Available tools

- **resume()** - Extracts and formats your PDF resume to markdown
- **fetch(url)** - Downloads and simplifies web content
- **validate()** - Returns configured phone number

## How resume formatting works

1. Reads PDF and extracts text + embedded links
2. Sends everything to OpenAI with formatting instructions
3. Returns clean markdown following a specific structure
4. Saves output to `resume.md`

## File structure

```
├── puch_mcp_server.py    # Main server code
├── .env                  # Environment variables (create from .env.example)
├── .env.example          # Template for environment setup
├── resume.pdf  # Your PDF resume
└── resume.md             # Generated markdown output
```

## Notes

- Requires OpenAI API access
- PDF must be text-readable (not image-based)
- Links are automatically mapped to appropriate resume sections (Might not be 100% accurate)
- Server uses bearer token authentication
