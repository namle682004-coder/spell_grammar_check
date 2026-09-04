from sqlalchemy import Column, String, Integer, Float, Text, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from src.storage.base import Base, TimestampMixin, generate_id
import enum
import boolean

class CorrectionType(enum.Enum):
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    STYLE = "style"
    TYPO = "typo"

class CorrectionDetail(Base, TimestampMixin):
    __tablename__ = "correction_details"
    
    id = Column(String(16), primary_key=True, default=generate_id)
    request_id = Column(String(16), ForeignKey("spell_grammar_requests.id", ondelete="CASCADE"), nullable=False)
    
    # Correction info
    correction_type = Column(Enum(CorrectionType), nullable=False)
    original_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=False)
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    
    # Suggestion metadata
    suggestion_id = Column(String(255), nullable=True)  # ID from LLM or rule engine
    confidence = Column(Float, nullable=True)  # 0-1
    rule_id = Column(String(100), nullable=True)  # Grammar rule ID if from rule engine
    
    # Context
    context_before = Column(Text, nullable=True)
    context_after = Column(Text, nullable=True)
    
    # Ignore/learn
    is_user_ignored = Column(boolean, default=False)
    is_learned = Column(boolean, default=False)  # User added to dictionary
    
    # Relationships
    request = relationship("SpellGrammarRequest", back_populates="correction_details")
    
    def to_dict(self):
        return {
            "type": self.correction_type.value,
            "original": self.original_text,
            "corrected": self.corrected_text,
            "position": {"start": self.start_position, "end": self.end_position},
            "confidence": self.confidence
        }