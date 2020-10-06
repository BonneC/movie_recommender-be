"""add all table

Revision ID: 898244526e4d
Revises: 
Create Date: 2020-09-24 01:31:34.454612

"""
from alembic import op
import sqlalchemy as sa

from crud import get_password_hash

# revision identifiers, used by Alembic.
revision = '898244526e4d'
down_revision = None
branch_labels = None
depends_on = None

# alembic revision --autogenerate -m "add all table"


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('movies',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('imdb_id', sa.String(), nullable=True),
    sa.Column('title', sa.String(), nullable=True),
    sa.Column('release_date', sa.Date(), nullable=True),
    sa.Column('vote_average', sa.Float(), nullable=True),
    sa.Column('vote_count', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    users = op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(), nullable=True),
    sa.Column('password', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('ratings',
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('movie_id', sa.Integer(), nullable=False),
    sa.Column('rating', sa.Float(), nullable=True),
    sa.Column('timestamp', sa.Date(), nullable=True),
    sa.ForeignKeyConstraint(['movie_id'], ['movies.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('user_id', 'movie_id')
    )
    op.bulk_insert(
        users,
        [
            {'username': 'jas', 'password': get_password_hash('1234')},
            {'username': 'tis', 'password': get_password_hash('sotiegajle')},
        ])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('ratings')
    op.drop_table('users')
    op.drop_table('movies')
    # ### end Alembic commands ###