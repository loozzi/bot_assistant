from typing import Literal

from pydantic import BaseModel

CategoryGroup = Literal["essential", "lifestyle", "savings_debt", "other"]


class CategoryParams(BaseModel):
    action: Literal["list", "create", "rename", "merge", "delete"]
    # create: the new name; rename/merge/delete: the existing category acted on
    name: str | None = None
    # rename: the new name; merge: the category `name` is merged INTO
    target_name: str | None = None
    group: CategoryGroup | None = None
