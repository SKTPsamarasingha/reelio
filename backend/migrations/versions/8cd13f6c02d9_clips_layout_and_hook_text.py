"""clips layout and hook text

Revision ID: 8cd13f6c02d9
Revises: 2754586d4032
Create Date: 2026-11-12 17:24:38.180591

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8cd13f6c02d9'
down_revision: Union[str, Sequence[str], None] = '2754586d4032'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("clips", "platform", new_column_name="layout")
    op.add_column("clips", sa.Column("hook_text", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("clips", "hook_text")
    op.alter_column("clips", "layout", new_column_name="platform")
