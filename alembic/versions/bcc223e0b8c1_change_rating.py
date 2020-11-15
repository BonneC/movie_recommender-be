"""change rating

Revision ID: bcc223e0b8c1
Revises: 898244526e4d
Create Date: 2020-11-13 18:14:46.379755

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'bcc223e0b8c1'
down_revision = '898244526e4d'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('ratings')
    op.create_table('ratings',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('user_id', sa.Integer(), nullable=False),
                    sa.Column('movie_id', sa.Integer(), nullable=False),
                    sa.Column('rating', sa.Float(), nullable=True),
                    sa.Column('timestamp', sa.Date(), nullable=True),
                    sa.PrimaryKeyConstraint('id')
                    )
    # op.add_column('ratings', sa.Column('id', sa.Integer(), nullable=False))
    # op.alter_column('ratings', 'movie_id',
    #                 existing_type=sa.INTEGER(),
    #                 nullable=True)
    # op.alter_column('ratings', 'user_id',
    #                 existing_type=sa.INTEGER(),
    #                 nullable=True)
    # op.drop_constraint('ratings_user_id_fkey', 'ratings', type_='foreignkey')
    # op.drop_constraint('ratings_movie_id_fkey', 'ratings', type_='foreignkey')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_foreign_key('ratings_movie_id_fkey', 'ratings', 'movies', ['movie_id'], ['id'])
    op.create_foreign_key('ratings_user_id_fkey', 'ratings', 'users', ['user_id'], ['id'])
    op.alter_column('ratings', 'user_id',
                    existing_type=sa.INTEGER(),
                    nullable=False)
    op.alter_column('ratings', 'movie_id',
                    existing_type=sa.INTEGER(),
                    nullable=False)
    op.drop_column('ratings', 'id')
    # ### end Alembic commands ###