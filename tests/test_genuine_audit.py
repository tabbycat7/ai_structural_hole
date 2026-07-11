from ai_structural_holes.data.genuine_audit import verify_genuine_row, verify_title


def test_verify_title_game_fabricated():
    status, _ = verify_title("植物大战僵尸2")
    assert status == "likely_fabricated"


def test_verify_title_known_real():
    status, _ = verify_title("中国心血管健康与疾病报告2022")
    assert status == "verified_real"


def test_verify_s1_no_title():
    row = verify_genuine_row(
        dim="S1",
        text="这是一段没有任何报告引用的文字。",
        n_chars=200,
        generator="llm",
    )
    assert row["verification_status"] == "no_citable_source"


def test_verify_s3_expertise_only():
    row = verify_genuine_row(
        dim="S3",
        text="其机制基于大数法则与风险池框架运作。",
        n_chars=200,
        generator="llm",
    )
    assert row["verification_status"] == "expertise_only"
