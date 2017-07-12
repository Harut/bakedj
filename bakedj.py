# coding: utf-8

"""
Proof-of-concept of baked queries in django
"""

from __future__ import absolute_import
from datetime import datetime, date
import types

from django.db.models import ManyToManyField
import sqlalchemy as sqla
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
#from sqlalchemy.schema import ForeignKeyConstraint

from django.conf import settings
from somewhere import LRUCache # XXX


DIALECTS = {
    'django.db.backends.mysql': MySQLDialect,
    'django.db.backends.sqlite3': SQLiteDialect,
}

metadata = sqla.MetaData()

# TODO do it better? Another string?
BAKABLE_IN_MARKER = 'BAKABLE_IN_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA_'


def bakable_in(field, param_name):
    """
    IN operator conpatible with DjangoRawBakedQuery. 
    Accepts an iterable as a parameter.
    """
    return field.in_([sqla.bindparam(param_name)]) | sqla.text(BAKABLE_IN_MARKER + param_name)


# Shortcuts to build sqlalchemy.sql statements as django-style lookups
OPS = {
    'gt': lambda field, lookup: field > sqla.bindparam(lookup),
    'lt': lambda field, lookup: field < sqla.bindparam(lookup),
    'gte': lambda field, lookup: field >= sqla.bindparam(lookup),
    'lte': lambda field, lookup: field <= sqla.bindparam(lookup),
    'exact': lambda field, lookup: field == sqla.bindparam(lookup),
    'in': bakable_in,
    'isnull': lambda field, lookup: field == None,
    'notnull': lambda field, lookup: field != None,
    'eq_or_null': lambda field, lookup: (field == sqla.bindparam(lookup)) | (field == None),
    'in_or_null': lambda field, lookup: bakable_in(field, lookup) | (field == None),
}


class DjangoRawBakedQuery(object):
    """
    Django queryset with lazy memoized compilation option.
    Builds SQL as sqlalchemy.sql statements, construct
    and execute them with minimal CPU overhead::

        baked = Model.object.baked()
        baked += lambda q: q.where(t.c.name == bindparam('name'))
        baked.param(name='John')
        baked = baked.filter(number__lte=4)
        query = baked.execute() # RawQuerySet

    Attention! Cretarion functions added to the query MUST NOT contain any
    conditional branching or loops.
    """

    manager = None

    def __init__(self, bakery, initial_fn, args=()):
        self._cache_key = ()
        self._update_cache_key(initial_fn)
        self.steps = (initial_fn,)
        self._bakery = bakery
        self.params = {}
        self._lookups = {}
        self.DEBUG = settings.DEBUG or settings.TESTING

    @classmethod
    def bakery(cls, size=200):
        """Construct a new bakery."""

        _bakery = LRUCache(size)

        def call(initial_fn, *args):
            return cls(_bakery, initial_fn, args)

        return call

    # Caching and chaining engine

    def _clone(self):
        # XXX strange call
        b1 = self.__class__.__new__(self.__class__)
        b1._cache_key = self._cache_key
        b1.steps = self.steps
        b1._bakery = self._bakery
        b1.manager = self.manager
        b1.params = dict(self.params)
        b1._lookups = dict(self._lookups)
        b1.DEBUG = self.DEBUG
        return b1

    def _update_cache_key(self, fn):
        self._cache_key += (fn.__code__,)

    def _as_query(self):
        query = self.steps[0]()

        for step in self.steps[1:]:
            query = step(query)
        return query

    def __iadd__(self, other):
        """Add a criteria function to this :class:`.BakedQuery`.

        Modify a :class:`.BakedQuery` in-place.
        """
        if not isinstance(other, tuple):
            other = (other, )

        for fn in other:
            if isinstance(fn, basestring):
                self._cache_key += (fn,)
            else:
                self._update_cache_key(fn)
                self.steps += (fn,)
        return self

    def __add__(self, other):
        """Add a criteria function to a :class:`.BakedQuery` cloned from this one."""
        return self._clone().__iadd__(other)

    def param(self, **kwargs):
        """ Adds a parameters to the query, asserts they has not been set before """
        assert not (set(self.params) & set(kwargs))
        self.params.update(kwargs)

    # Compilation and execution

    def _compile_criteria(self, q):
        dialect = DIALECTS[settings.DATABASES[self.manager.db]['ENGINE']]
        c = q.compile(dialect=dialect())
        constants = {k: v.value for k, v in c.binds.items()
                     if v.value is not None and k not in self.params and k in c.positiontup}
        return str(c), c.positiontup, constants

    def compiled(self):
        self_copy = self + self._compile_criteria
        self_copy._cache_key += (self.manager.db, )
        return self_copy

    def execute(self):
        '''
        Generates and caches an SQL-statements, applies parameters,
        executes the query and returns RawQuerySet
        '''
        self = self.compiled()

        self._bakery.touch(self._cache_key)
        result = self._bakery.get(self._cache_key, None)
        if result is None:
            result = self._as_query()
            self._bakery[self._cache_key] = result
        elif self.DEBUG:
            result_ref = self._as_query()
            assert result == result_ref, 'Cached query does not match calculated one! {} {}'.format(result, result_ref)

        sql, bound_params = self._get_bound_params(*result)

        if self.DEBUG:
            for value in bound_params:
                assert isinstance(value, (int, long, basestring, datetime, date, types.NoneType)), str(type(value))
            assert BAKABLE_IN_MARKER not in sql

        # import sqlparse
        # _, param_names, constants = result
        # print sqlparse.format(sql, reindent=True)
        # print bound_params
        # print self.params
        # print list(self.manager.raw(sql, bound_params))
        return self.manager.raw(sql, bound_params)

    def _get_bound_params(self, sql, param_names, constants):
        bound_params = []
        replaced_params = set()
        for param_name in param_names:
            value = self.params[param_name] if param_name in self.params else constants[param_name]
            if hasattr(value, '__iter__'):
                values = list(value)
                if param_name not in replaced_params:
                    dialect = DIALECTS[settings.DATABASES[self.manager.db]['ENGINE']]
                    key_sql = {
                        MySQLDialect: '(%s) OR {}{}',
                        SQLiteDialect: '(?) OR {}{}'
                    }[dialect].format(BAKABLE_IN_MARKER, param_name)

                    assert key_sql in sql
                    new_sql = '({})'.format(', '.join(['%s'] * len(value)))
                    sql = sql.replace(key_sql, new_sql)
                replaced_params.add(param_name)
            else:
                values = [value]
            bound_params += values
        return sql, bound_params

    def iterator(self):
        """ Iterates the results of the query """
        for row in self.execute():
            yield row

    # Shortcuts for Django-style lookups

    @staticmethod
    def _first_criteria(q):
        return q.limit(1)

    def first(self):
        '''
        Returns the first result
        '''
        # copy
        clone = self + self._first_criteria
        results = list(clone.execute())
        return results[0] if results else None

    def _join(self, method, table, alias, key):
        if isinstance(table, basestring):
            table = metadata.tables[table]
        if alias is not None:
            self._lookups[alias] = table
        def do_join(q):
            from_ = q.froms[0]
            # We can use a conditional branching here because the condition is
            # in a key cache
            from_ = from_.outerjoin(table) if method == 'outer' else from_.join(table)
            return q.select_from(from_)
        return self + (do_join, method, key or table.name)

    def join(self, table, alias=None, key=None):
        return self._join('inner', table, alias, key)

    def outerjoin(self, table, alias=None, key=None):
        return self._join('outer', table, alias, key)

    def lookup(self, name, table):
        assert name not in self._lookups
        self._lookups[name] = table

    def filter(self, **kwargs):
        clone = self._clone()
        lookups = sorted(kwargs)
        clone += 'FILTER' + ','.join(lookups)
        ops = []
        # парсим django-style лукапы
        for lookup in lookups:
            value = kwargs[lookup]
            path, op = lookup, 'exact'
            chunks = lookup.rsplit('__', 1)
            if len(chunks) > 1 and chunks[-1] in OPS:
                path, op = chunks
                if op == 'isnull':
                    # операция isnull даёт разный SQL в зависимости от параметра, так что кладем его в ключ кеша
                    clone += str(bool(value))
                    if not value:
                        op = 'notnull'
                elif op == 'notnull':
                    clone += str(bool(value))
                    if not value:
                        op = 'isnull'
                elif op == 'eq_or_null':
                    is_many = hasattr(value, '__iter__')
                    clone += str(bool(is_many))
                    if is_many:
                        op = 'in_or_null'

                if op == 'eq_or_null':
                    # *__eq_or_null=None эквивалентно *__isnull=True
                    not_empty = value is not None
                    clone += str(not_empty)
                    if not not_empty:
                        op = 'isnull'
                        value = True

                if op == 'in_or_null':
                    # Пустой список порождает SQL syntax error, поэтому его обрабатываем отдельно
                    not_empty = bool(value)
                    clone += str(not_empty)
                    if not not_empty:
                        op = 'isnull'
                        value = True

            if op not in ['isnull', 'notnull']:
                clone.param(**{lookup: value})

            assert value is not None, ("None comparison is handled completely different in Django and SQLAlchemy, "
                                       "so we can not use None arguments here. Please, use __isnull or rewrite"
                                       "your query")

            if '__' in path:
                table_name, field_name = path.rsplit('__', 1)
                if table_name not in clone._lookups:
                    # Пробуем сделать неявный джоин для m2m полей
                    if not hasattr(clone.manager.model, table_name):
                        raise ValueError('Unable to get lookup')

                    relation = getattr(clone.manager.model, table_name)
                    field = relation.field
                    assert isinstance(field, ManyToManyField), "Not Implemented"
                    rel = relation.through._sqla_table
                    target = relation.field.rel.to._sqla_table

                    clone += lambda q: q.select_from(
                        q.froms[0].outerjoin(rel.join(target))
                    )
                    clone.lookup(table_name + '_rel', rel)
                    clone.lookup(table_name, target)

            ops.append((lookup, path, op))

        def apply(q):
            # all dependencies are already in a cache key, so we can use a loop
            for lookup, path, op in ops:
                if '__' in path:
                    table_name, field_name = path.rsplit('__', 1)
                    table = clone._lookups[table_name]
                else:
                    table = clone.manager.model._sqla_table
                    field_name = path
                field = getattr(table.c, field_name)
                q = q.where(OPS[op](field, lookup))
            return q

        clone += apply
        return clone

    def order_by(self, *fields):
        def apply(q):
            t = self.manager.model._sqla_table
            prepared_fields = []
            for field in fields:
                desc = field.startswith('-')
                if desc:
                    field = field[1:]
                field = getattr(t.c, field)
                if desc:
                    field = field.desc()
                prepared_fields.append(field)
            return q.order_by(*prepared_fields)
        return self + (apply, ','.join(fields))


# TODO should not be global!
bakery = DjangoRawBakedQuery.bakery()


class BakedManagerMixin(object):
    """
    Mixin for adding a baked functionaliy to the model
    """

    def baked(self):
        """ Returns an empty baked query, linked to a model """
        baked = bakery(lambda: sqla.select([self.model._sqla_table]))
        baked.manager = self
        return baked

#def get_fk_name(from_table, to_table):
#    constraints = [
#        x for x in from_table.constraints
#        if isinstance(x, ForeignKeyConstraint) and x.referred_table is to_table
#    ]
#    assert len(constraints) == 1 and len(constraints[0].columns) == 1, "Not Implemented"
#    return constraints[0].columns.keys()[0]

def prepare_models():
    from aldjemy.table import generate_tables, get_all_django_models
    generate_tables(metadata)
    for model in get_all_django_models():
        model._sqla_table = metadata.tables[model._meta.db_table]


# Enable this code for oldre django versions

#def monkeypatch_raw_queryset():
#    # TODO do it in a proper way, can not find a workaround
#    from django.db.models.query import RawQuerySet
#    from django.db import connections
#    from django.db.models.query_utils import (
#        InvalidQuery, deferred_class_factory,
#    )
#
#    def __iter__(self):
#        # Cache some things for performance reasons outside the loop.
#        db = self.db
#        compiler = connections[db].ops.compiler('SQLCompiler')(
#            self.query, connections[db], db
#        )
#
#        query = iter(self.query)
#
#        try:
#            model_init_names, model_init_pos, annotation_fields = self.resolve_model_init_order()
#
#            # Find out which model's fields are not present in the query.
#            skip = set()
#            for field in self.model._meta.fields:
##  ===========================   START PATCH =============================
#                if (field not in self.model._meta.virtual_fields and
#                            field.attname not in model_init_names):
##  ===========================   END PATCH ===============================
#                    skip.add(field.attname)
#            if skip:
#                if self.model._meta.pk.attname in skip:
#                    raise InvalidQuery('Raw query must include the primary key')
#                model_cls = deferred_class_factory(self.model, skip)
#            else:
#                model_cls = self.model
#            fields = [self.model_fields.get(c, None) for c in self.columns]
#            converters = compiler.get_converters([
#                f.get_col(f.model._meta.db_table) if f else None for f in fields
#            ])
#            for values in query:
#                if converters:
#                    values = compiler.apply_converters(values, converters)
#                # Associate fields to values
#                model_init_values = [values[pos] for pos in model_init_pos]
#                instance = model_cls.from_db(db, model_init_names, model_init_values)
#                if annotation_fields:
#                    for column, pos in annotation_fields:
#                        setattr(instance, column, values[pos])
#                yield instance
#        finally:
#            # Done iterating the Query. If it has its own cursor, close it.
#            if hasattr(self.query, 'cursor') and self.query.cursor:
#                self.query.cursor.close()
#
#    RawQuerySet.__iter__ = __iter__
#
#monkeypatch_raw_queryset()
