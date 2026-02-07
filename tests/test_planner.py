from rg_atp_pipeline.planner import plan_new, plan_old, plan_new_docs, plan_old_docs


def test_plan_new_includes_res_and_rg() -> None:
    urls = plan_new("https://example.com/base", 2024, 1, 1)
    assert urls == [
        "https://example.com/base/res-2024-1-20-1.pdf",
        "https://example.com/base/rg-2024-1-20-1.pdf",
    ]

    docs = plan_new_docs("https://example.com/base", 2024, 1, 1)
    assert [doc.doc_key for doc in docs] == ["res-2024-1-20-1", "rg-2024-1-20-1"]
    assert [doc.url for doc in docs] == [
        "https://example.com/base/res-2024-1-20-1.pdf",
        "https://example.com/base/rg-2024-1-20-1.pdf",
    ]


def test_plan_old_includes_year_suffixes() -> None:
    urls = plan_old("https://example.com/base", 1, 1, 1, 2023, 2023)
    assert urls == [
        "https://example.com/base/1.pdf",
        "https://example.com/base/1-2023.pdf",
        "https://example.com/base/1-23.pdf",
    ]

    docs = plan_old_docs("https://example.com/base", 1, 1, 1, 2023, 2023)
    assert [doc.doc_key for doc in docs] == ["OLD-1", "OLD-1-2023", "OLD-1-23"]
    assert [doc.url for doc in docs] == [
        "https://example.com/base/1.pdf",
        "https://example.com/base/1-2023.pdf",
        "https://example.com/base/1-23.pdf",
    ]
