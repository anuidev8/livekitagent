from knowledge_base import search_seti_knowledge


def test_generic_company_question_returns_real_seti_content() -> None:
    # A maximally generic query has little lexical signal to rank on — any
    # topically relevant chunk is an acceptable answer for the voice model
    # to compose from, so only assert it's genuine, on-topic content.
    result = search_seti_knowledge("¿qué es SETI y a qué se dedica?")
    assert "SETI" in result


def test_specific_identity_question_surfaces_the_identity_chunk() -> None:
    result = search_seti_knowledge("¿cuál es la razón social de SETI?")
    assert "Servicios Especializados de Tecnología" in result


def test_client_question_surfaces_client_list() -> None:
    result = search_seti_knowledge("¿qué bancos son clientes de SETI?")
    assert "Bancolombia" in result


def test_contact_question_surfaces_contact_channel() -> None:
    result = search_seti_knowledge("¿cómo puedo contactar a SETI comercialmente?")
    assert "comercial@seti.com.co" in result


def test_unmatched_query_returns_safe_fallback_instead_of_crashing() -> None:
    result = search_seti_knowledge("xyzzy quantum banana unrelated nonsense query")
    assert isinstance(result, str)
    assert result  # never empty — the agent must always get something to speak from


def test_broad_conversational_query_grounds_in_core_identity() -> None:
    """Regression test for a real production session (2026-09-02 logs):

    visitor asked "y me cuentas de la empresa seti qué más sabes de seti" —
    the old ranking buried the rich Identidad chunk behind a contact-emails
    chunk (a stemming collision: "cuentas" -> "cuenta", matching "cuenta de
    respaldo institucional") and a thin one-line client-portfolio fragment.
    Nova ended up narrating the tool's perceived gaps instead of speaking
    confidently. The identity/value-prop chunks must always be present so
    broad company questions have solid material regardless of keyword noise.
    """
    result = search_seti_knowledge(
        "y me cuentas de la empresa seti qué más sabes de seti"
    )
    assert "Servicios Especializados de Tecnología" in result
    assert (
        "Propuesta de valor central" in result
        or "Apoyar a las organizaciones" in result
    )


def test_excluded_methodology_domain_never_appears_in_the_index() -> None:
    """Guards the domain split: SETI-Spec/OpenSpec/SDD must never leak into the
    kiosk's knowledge base, even if data/ is edited later without re-reading
    KNOWLEDGE_BASE_NOTES.md."""
    from knowledge_base import _load_chunks

    all_text = " ".join(chunk.text.lower() for chunk in _load_chunks())
    for forbidden in ("openspec", "seti-spec", "sdd", "vibe coding", "código zombi"):
        assert forbidden not in all_text, f"methodology domain leaked in: {forbidden!r}"
