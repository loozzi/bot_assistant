from pydantic import BaseModel, Field


class ResearchSummary(BaseModel):
    """Structured output from the summarize-and-save LLM call."""
    answer: str
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Render as a formatted string for the Telegram reply."""
        lines = [self.answer]

        if self.key_points:
            lines.append("")
            for point in self.key_points:
                lines.append(f"• {point}")

        return "\n".join(lines)
