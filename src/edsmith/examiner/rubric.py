"""IELTS Task 2 band descriptors for all four components (bands 4–9)."""
from __future__ import annotations

_DESCRIPTORS: dict[str, dict[int, str]] = {
    "task_response": {
        9: "Fully addresses all parts; presents a fully developed position with well-extended and well-supported ideas.",
        8: "Sufficiently addresses all parts; presents and supports main ideas; well-extended position, though may occasionally over-generalise.",
        7: "Addresses all parts; presents a clear position throughout; extends and supports main ideas but may over-generalise; some ideas may lack full development.",
        6: "Addresses all parts though some more fully than others; relevant position but conclusions may become unclear; presents relevant main ideas but some are inadequately developed.",
        5: "Addresses the task only partially; format may be inappropriate; position expressed but development not always clear; main ideas are limited and not sufficiently developed; may include irrelevant detail.",
        4: "Responds minimally or tangentially; unclear position; main ideas difficult to identify, often repetitive or unsupported; format may be inappropriate.",
    },
    "coherence": {
        9: "Cohesion is invisible; paragraphing is skilfully managed; information and ideas flow logically throughout.",
        8: "Information and ideas are sequenced logically; all aspects of cohesion are managed well; paragraphing is used sufficiently and appropriately.",
        7: "Clear overall progression; logically organised; range of cohesive devices used appropriately; each paragraph has a clear central topic.",
        6: "Clear overall progression; cohesive devices used effectively but cohesion within or between sentences may be faulty or mechanical; paragraphing used, though not always logically.",
        5: "Some organisation but may lack overall progression; cohesive devices inadequate, inaccurate, or overused; may be repetitive due to poor referencing; paragraphs lack clear central topic.",
        4: "Information and ideas not arranged coherently; basic cohesive devices used but may be inaccurate or repetitive; paragraphing may be absent or inadequate.",
    },
    "lexical": {
        9: "Wide range of vocabulary used with very natural and sophisticated control; rare minor errors occur only as slips.",
        8: "Wide range of vocabulary used fluently and flexibly for precise meanings; uncommon items used skilfully; rare errors in word choice, collocation, spelling or word formation.",
        7: "Sufficient range of vocabulary for flexibility and precision; less common items used with some awareness of style; occasional errors in word choice, spelling or word formation, but these do not impede communication.",
        6: "Adequate range of vocabulary for the task; attempts less common vocabulary but with some inaccuracy; errors in spelling or word formation do not impede communication.",
        5: "Limited range of vocabulary but minimally adequate; noticeable errors in spelling or word formation may cause some difficulty for the reader.",
        4: "Only basic vocabulary, often repetitive or inappropriate; limited control of word formation and spelling; errors may cause strain for the reader.",
    },
    "grammar": {
        9: "Wide range of structures used with full flexibility and accuracy; rare minor errors occur only as slips.",
        8: "Wide range of structures; majority of sentences are error-free; only very occasional errors or inappropriacies.",
        7: "Variety of complex structures; frequent error-free sentences; good control of grammar and punctuation with only a few errors.",
        6: "Mix of simple and complex sentence forms; errors in grammar and punctuation are present but rarely reduce communication.",
        5: "Limited range of structures; complex sentences attempted but less accurate than simple ones; frequent grammatical errors; punctuation may be faulty; errors can cause difficulty for the reader.",
        4: "Very limited range of structures; some structures are accurate but errors predominate; punctuation often faulty.",
    },
}

_COMPONENT_NAMES = {
    "task_response": "Task Achievement",
    "coherence": "Coherence and Cohesion",
    "lexical": "Lexical Resource",
    "grammar": "Grammatical Range and Accuracy",
}


def get_band_descriptors(component: str, bands: tuple[int, ...] = (4, 5, 6, 7, 8, 9)) -> str:
    """Return a formatted rubric string for the given component and band range."""
    table = _DESCRIPTORS.get(component, {})
    name = _COMPONENT_NAMES.get(component, component)
    lines = [f"IELTS Band Descriptors — {name}:"]
    for band in bands:
        desc = table.get(band, "")
        if desc:
            lines.append(f"  Band {band}: {desc}")
    return "\n".join(lines)
