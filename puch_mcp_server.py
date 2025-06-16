from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from openai import BaseModel, OpenAI
from pydantic import AnyUrl, Field
import readabilipy
from pathlib import Path
import os
from dotenv import load_dotenv
import pypdf

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("TOKEN", "<generated_token>") 
MY_NUMBER = os.getenv("MY_NUMBER", "9179XXXXXXXX") 
print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("TOKEN"))
print(os.getenv("MY_NUMBER"))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    For a more complete implementation that can authenticate dynamically generated tokens,
    please use `BearerAuthProvider` with your public key or JWKS URI.
    """

    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,  # No expiration for simplicity
            )
        return None


class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        """
        Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
        """
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content


def extract_links_from_page(page):
    """Extract hyperlinks and annotations from a PDF page."""
    links = []
    
    try:
        # Check if page has annotations
        if "/Annots" in page:
            annotations = page["/Annots"]
            for annotation in annotations:
                annotation_obj = annotation.get_object()
                
                # Check if it's a link annotation
                if annotation_obj.get("/Subtype") == "/Link":
                    # Get the action associated with the link
                    if "/A" in annotation_obj:
                        action = annotation_obj["/A"]
                        if "/URI" in action:
                            uri = action["/URI"]
                            # Get the rectangle coordinates for positioning
                            rect = annotation_obj.get("/Rect", [])
                            links.append({
                                "uri": str(uri),
                                "rect": rect
                            })
    except Exception as e:
        print(f"Warning: Could not extract links from page: {e}")
    
    return links


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content and hyperlinks from a PDF file and return it as a string.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content with hyperlinks information
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            all_links = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                # Extract links/annotations from the page
                links = extract_links_from_page(page)
                all_links.extend(links)
                
                # Add page text
                text += page_text + "\n"
            
            # Append all found links at the end
            if all_links:
                text += "\n\n-- EXTRACTED LINKS AND URLS --\n"
                
                # Add annotation links
                if all_links:
                    text += "\nHyperlinks from PDF annotations:\n"
                    for i, link in enumerate(all_links, 1):
                        text += f"{i}. {link['uri']}\n"

            
            return text.strip()
    except Exception as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR, 
                message=f"Failed to extract text from PDF: {e!r}"
            )
        )

def format_resume_with_openai(extracted_text: str) -> str:
    """
    Use OpenAI to format the extracted resume text according to the specific resume.md format.
    
    Args:
        extracted_text: Raw text extracted from the PDF
        
    Returns:
        Formatted markdown resume
    """
    try:
        prompt = f"""
            Please format the following resume text into clean markdown following this EXACT format and structure.

            CRITICAL: The text contains extracted hyperlinks and URLs that MUST be properly mapped and integrated into the resume. DO NOT create a separate "Links Found" section.

            # [Full Name]

            **Phone:** [phone number]  
            **Email:** [email in markdown link format]  
            **Portfolio:** [portfolio link]  
            **LinkedIn:** [linkedin link]  
            **GitHub:** [github link]  
            **LeetCode:** [leetcode link if available]  
            **HackerRank:** [hackerrank link if available]

            ---

            ## Education

            **[Institution Name]** | [Years]  
            [Degree and Major]

            ---

            ## Technical Skills

            **Programming Languages:** [list]  
            **Databases:** [list]  
            **Frameworks and Tools:** [list]  
            **Developer Tools:** [list]  
            **Relevant Coursework:** [list]  
            **Soft Skills:** [list]

            ---

            ## Experience

            **[Company Name]** | [Dates]  
            _[Position Title]_ | [Location]

            • [Achievement/responsibility 1]  
            • [Achievement/responsibility 2]  
            • [Achievement/responsibility 3]  

            ---

            ## Projects

            **[Project Name]**  
            _Technologies:_ [tech stack]  
            _GitHub:_ [github link if available]

            • [Description point 1]  
            • [Description point 2]  
            • [Description point 3]

            **Other Projects:**
            - **[Project Name]:** [link]
            - **[Project Name]:** [link]

            ---

            ## Volunteering Experience

            **[Organization Name]** | [Dates]  
            _[Position Title]_

            • [Achievement/responsibility 1]  
            • [Achievement/responsibility 2]

            ---

            ## Certifications/Courses

            • **[Course Name]** - [Provider] ([Platform]) | [Certificate link if available]  
            • **[Course Name]** - [Provider] ([Platform]) | [Certificate link if available]

            ---

            ## Achievements

            • **[Achievement level]** [description]  
            • **[Achievement level]** [description]  
            • **[Achievement level]** [description]

            CRITICAL LINK MAPPING INSTRUCTIONS:
            1. The text contains a section "=== EXTRACTED LINKS AND URLS ===" with all found links
            2. You MUST intelligently map these links to the appropriate sections:
            - Email links go in the header
            - GitHub links go with projects and in the header
            - LinkedIn links go in the header
            - Portfolio/personal website links go in the header
            - Certificate links go with certifications
            - Project links go with their respective projects
            3. DO NOT include the "=== EXTRACTED LINKS AND URLS ===" section in the final output
            4. DO NOT create a separate "Links Found" section
            5. Ensure ALL links are properly integrated into their relevant sections

            FORMATTING RULES:
            1. Use underscores (_text_) for italics, NOT asterisks
            2. Use **text** for bold
            3. Use horizontal dividers (---) between sections
            4. Use • for bullet points
            5. Include proper markdown links [text](url)
            6. Ensure all URLs have proper protocols (https://)
            7. Clean up any malformed or duplicate links
            8. Map GitHub repository links to their respective projects
            9. Map certification links to their respective courses
            10. Keep consistent spacing and formatting

            Here is the raw resume text with extracted links to format:

            {extracted_text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at formatting resumes into clean, professional markdown. Follow the exact format specified in the prompt."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Failed to format resume with OpenAI: {e!r}"
            )
        )


mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, \
no extra formatting.",
    side_effects=None,
)

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text.
    
    This function:
    1. Finds and reads your resume PDF file.
    2. Extracts text content from the PDF.
    3. Uses OpenAI to format the resume according to the specific markdown format.
    4. Returns the resume as markdown text.
    """
    try:
        # Path to the resume PDF file
        resume_path = "resume.pdf"
        
        # Check if the file exists
        if not Path(resume_path).exists():
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Resume file not found: {resume_path}"
                )
            )
        
        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(resume_path)
        
        if not extracted_text.strip():
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message="No text could be extracted from the resume PDF"
                )
            )
        print(extracted_text)
        print("--------------------------------")
        
        # Format with OpenAI according to resume.md format
        markdown_resume = format_resume_with_openai(extracted_text)
        
        # Save the formatted resume
        with open("resume.md", "w", encoding="utf-8") as f:
            f.write(markdown_resume)
        print(markdown_resume)
        return markdown_resume
        
    except Exception as e:
        # Handle any unexpected errors
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Error processing resume: {e!r}"
            )
        )


@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER


FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)


@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            # Only add the prompt to continue fetching if there is still remaining content
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]


async def main():
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 