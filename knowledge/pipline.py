
import logging
from .generator import generate_knowledge, regenerate_knowledge
from .embedding import generate_embedding
from .schemas import (
    ApprovedKnowledge,
    EntityType,
    GeneratedKnowledge,
    KnowledgeGenerationRequest,
    VectorMetadata,
)
from .updater import update_knowledge
from .normalizer import normalize_knowledge
from .vector_store import upsert_vector

logger = logging.getLogger(__name__)



def run_pre_approval_stage(
    request: KnowledgeGenerationRequest,
) -> GeneratedKnowledge:
    generated = generate_knowledge(request)
    generated = normalize_knowledge(generated, request.name)   # ← بعت الاسم هنا
    return generated


def regenerate(
    request: KnowledgeGenerationRequest,
    previous_output: GeneratedKnowledge,
    admin_feedback: str | None = None,
) -> GeneratedKnowledge:
    generated = regenerate_knowledge(
        request=request,
        previous_output=previous_output,
        admin_feedback=admin_feedback,
    )
    generated = normalize_knowledge(generated, request.name)   # ← وهنا كمان
    return generated


def run_post_approval_stage(entity_id: int, entity_type: EntityType, name: str,
                             final_knowledge: GeneratedKnowledge) -> None:
    """
    Runs after the admin clicks "Approve":
        Update PostgreSQL -> Generate Embedding -> Insert into Vector Database
    """
    approved = ApprovedKnowledge(
        entity_id=entity_id,
        entity_type=entity_type,
        **final_knowledge.model_dump(),
    )

    update_knowledge(approved)
    embedding = generate_embedding(name, approved)
    metadata = VectorMetadata(id=entity_id, type=entity_type, name=name)
    upsert_vector(metadata, embedding)

    logger.info("Knowledge pipeline complete for %s id=%s ('%s').", entity_type.value, entity_id, name)