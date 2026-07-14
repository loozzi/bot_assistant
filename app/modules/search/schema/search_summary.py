from pydantic import BaseModel, Field


class SearchSource(BaseModel):
    """A single source document from search results."""
    title: str
    url: str


class SearchSummary(BaseModel):
    """Structured output from the summary LLM call."""
    answer: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[SearchSource] = Field(default_factory=list)
    
    def render(self) -> str:
        """
        Render the summary as a formatted string for Telegram reply.
        
        Format: answer + bullet points + sources list
        """
        lines = [self.answer]
        
        if self.key_points:
            lines.append("")
            for point in self.key_points:
                lines.append(f"• {point}")
        
        if self.sources:
            lines.append("")
            lines.append("Nguồn:")
            for source in self.sources:
                lines.append(f"• [{source.title}]({source.url})")
        
        return "\n".join(lines)
