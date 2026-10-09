from app import taxonomy
from app.schemas import Category


def test_taxonomy_shape():
    leaves = taxonomy.leaves()
    assert 35 <= len(leaves) <= 50
    for lid, leaf in leaves.items():
        assert lid.count(".") == 1 and leaf.label and leaf.description
        assert isinstance(leaf.category, Category)
    for needed in ["gambling.promotion", "drugs.promotion", "health.cure_claim",
                   "religion.insult", "child.csam", "fraud.financial"]:
        assert taxonomy.is_leaf(needed)


def test_parent_matching_and_categories():
    assert taxonomy.covers("gambling", "gambling.promotion")
    assert taxonomy.covers("gambling.promotion", "gambling.promotion")
    assert not taxonomy.covers("drugs", "gambling.promotion")
    assert not taxonomy.covers("drugs.promotion", "gambling.promotion")
    assert taxonomy.is_target("gambling") and not taxonomy.is_leaf("gambling")
    assert taxonomy.category_of("religion.insult") == Category.religious_offence
    assert taxonomy.clean(["Gambling.Promotion", "bogus.leaf", "gambling.promotion"]) == ["gambling.promotion"]
