class NE:

    recipe = None

    def insert_colon_left(self, string, index):
        return string[:index] + '[' + string[index:]

    def insert_colon_right(self, string, index):
        return string[:index] + ']' + string[index:]

    def insert_label(self, string, index, label):
        return string[:index] + '|' + label + string[index:]

    def get_sentence_id(self):
        pass

    def get_id(self):
        return self.span["token_end"]

    def get_label(self):
        return self.span["label"].lower()

    def add_dep(self, pos):
        self.memberships.add(pos)

    def out(self):
        return NE.recipe["text"][self.span["start"]: self.span["end"]]

    def __init__(self, span):
        self.span = span
        self.memberships = set()

class Link:

    def add_link(self, relation_type, token_out):
        self.links.append((relation_type, token_out))

    def link_out(self):
        out = ""
        if self.links:
            for link in self.links:
                out += link[0] + " = " + link[1] + ", "
            out = out[:-2]
        return out
    def __init__(self):
        self.links = list()

class Or(Link):

    def add_the_or(self, or_):
        self.or_ = or_
        self.add_link("Or", or_)

    def get_the_or(self):
        return self.or_

    def __init__(self):
        self.or_ = None
        super().__init__()

class Join(Link):
    def add_the_join(self, join):
        self.join = join
        self.add_link("Join", join)

    def get_the_join(self):
        return self.join

    def __init__(self):
        self.join = None
        super().__init__()

class Modifier(Link):

    def add_modifier(self, modifying):
        self.modifying = modifying
        self.add_link("Modifier", modifying)

    def get_modifier(self):
        return self.modifying

    def __init__(self):
        self.modifying = None
        super().__init__()

class Entity(NE, Or):

    def add_hierarchy(self, relation):
        pass

    def add_prime(self, prime):
        self.primes.add(prime)

    def get_entity_type(self):
        if "ingr" in self.get_label():
            return "ingr"
        elif "tool" in self.get_label():
            return "tool"
        raise NotImplementedError

    def get_primes_(self):
        if self.primes:
            return ', '.join(self.primes)
        raise NotImplementedError

    def __init__(self, *args):
        super().__init__(*args)
        self.primes = set()
        self.primes_filled = False
        Or.__init__(self)

class EntityAux(NE, Modifier):

    def __init__(self, *args):
        super().__init__(*args)
        # MRO, call also the Modifier constructor
        Modifier.__init__(self)


class State(Entity, Modifier):

    def __init__(self, *args):
        super().__init__(*args)
        Modifier.__init__(self)


class Why(NE):

    def __init__(self, *args):
        super().__init__(*args)

class Code(NE, Or, Join, Modifier):

    def __init__(self, span):
        super().__init__(span)
        self.lu = list()
        Or.__init__(self)
        Join.__init__(self)
        Modifier.__init__(self)
