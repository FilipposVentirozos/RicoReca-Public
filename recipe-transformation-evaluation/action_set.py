import copy
import inspect
import itertools
import ne
import rapidfuzz as fz
import warnings
from collections import defaultdict
from nltk import pos_tag
import static_variables as stv
try:
    pos_tag(["test"])
except LookupError:
    import nltk
    nltk.download('averaged_perceptron_tagger')
from pprint import pprint
import logging
log = logging.getLogger(__name__)

class NotEntityMatched(Exception):
    def __init__(self, message, action_set=None):
        super().__init__(message)
        self.action_set = action_set

class ReversibleDict(dict):
    def add_item(self, key, value):
        self.__setitem__(key, value)
        self.__setitem__(value, key)


class ActionSet:
    recipe = None
    annotator_added = True
    action_sets = None

    def get_word(self):
        if self.anchor_code:
            return ActionSet.recipe["text"][self.anchor_code.span["start"]: self.anchor_code.span["end"]]
        return None

    def is_parent_or_child(self, span):
        """ Returns the relationship label and if the supplied it's a parent or a child
        :param span:
        :return: Str, Str
        """
        try:
            for relation in ActionSet.recipe["relations"]:
                if relation["head"] == span["token_end"] and relation["child"] == self.anchor_code.span["token_end"]:
                    return relation["label"], "parent"
                elif relation["child"] == span["token_end"] and relation["head"] == self.anchor_code.span["token_end"]:
                    return relation["label"], "child"
        except TypeError:
            return None

    def get_distance(self, span):
        """ Measures the distance in tokens between the span and the span from the anchor code.
        :param span:
        :return:
        """
        if self.anchor_code:
            if self.anchor_code.span["token_end"] > span["token_end"]:
                return self.anchor_code.span["token_start"] - span["token_end"]
            else:
                return span["token_start"] - self.anchor_code.span["token_end"]
        return None

    @staticmethod
    def get_idx(iteratable):
        for j in iteratable:
            if not iteratable.get(j, False):
                continue
            for x, action_set in enumerate(iteratable[j]):
                yield j, x, action_set

    @staticmethod
    def get_action_set_position(action_set_or_id):
        for i, j, action_set_pair in ActionSet.get_idx(ActionSet.action_sets):
            try:
                if action_set_pair.anchor_code.get_id() == action_set_or_id.anchor_code.get_id():
                    return tuple((i, j))
            except AttributeError:
                # In case it's the id
                if action_set_pair.anchor_code.get_id() == action_set_or_id:
                    return tuple((i, j))
    def get_my_position(self):
        for i, j, action_set in ActionSet.get_idx(ActionSet.action_sets):
            if action_set.anchor_code.get_id() == self.anchor_code.get_id():
                return tuple((i, j))

    def get_dependents_(self):
        """ Yield the Code names that the current Code is depended on

        :return:
        """
        if self.anchor_code:
            for relation in self.relations["depend"]:
                yield ActionSet.recipe["text"][relation["child_span"]["start"]: relation["child_span"]["end"]]

    def is_span(self, span):
        if self.anchor_code:
            if span["token_end"] == self.anchor_code.span["token_end"]:
                return True
        return False

    def is_in_relation(self, relation):
        if self.anchor_code:
            if relation["head"] == self.anchor_code.span["token_end"]:
                return "head"
            elif relation["child"] == self.anchor_code.span["token_end"]:
                return "child"
            else:
                return False
        return False

    def get_masters(self, member=True, depend=True, join_=True, or_=True):
        # Do not include the `Or` link, if it has its own dependencies or members
        for k, relations in self.relations.items():
            if k == "or" and not or_:
                continue
            elif k == "member" and not member:
                continue
            elif k == "depend" and not depend:
                continue
            elif k == "join" and not join_:
                continue
            for relation in relations:                
                try:
                    if self != ActionSet.action_sets[relation["dest"][0]][relation["dest"][1]]:
                        yield ActionSet.action_sets[relation["dest"][0]][relation["dest"][1]]
                except KeyError:
                    if self != ActionSet.action_sets[relation["start"][0]][relation["start"][1]]:
                        yield ActionSet.action_sets[relation["start"][0]][relation["start"][1]]

    def remove_unobedient_children(self, master):
        kwargs = {"member": False, "depend": False, "join_":False, "or_": False, "remove_parents_junction_nodes": False,
        "remove_self": False, "remove_or": False}
        master_, _ = list(self.recv_masters("get_master_of", masters=None, paths=None, **kwargs))
        for pos, _ in master_.items():
            master.pop(pos, None)
        return master

    def remove_parent_children(self, action_set, masters, paths=None):
        if not paths:
            paths = set()
        # Check whether it has any children that are not master of self
        for relation_ in action_set.relations["member"]:
            try:
                action_set_pair = ActionSet.action_sets[relation_["dest"][0]][relation_["dest"][1]]
                if action_set_pair == self or action_set_pair == action_set:
                    continue
                # Check whether the ActionSet is a depended under the same parent
                kwargs = {"member": False, "depend": True, "or_": True, "join_": True, "remove_parents_junction_nodes": False}
                masters_, _ = self.recv_masters("get_masters", masters=None, paths=None, **kwargs)
                if action_set_pair in list(masters_.values()):
                    continue
                try:
                    # print()
                    # print(self.anchor_code.out())
                    # print(masters[relation_["dest"]].anchor_code.out())
                    if masters[relation_["dest"]] not in paths:
                        paths.add(masters[relation_["dest"]])
                        masters = self.remove_parent_children(masters[relation_["dest"]], masters, paths=paths)
                except KeyError:
                    pass
                masters.pop(relation_["dest"], None)
            except KeyError:
                continue
        return masters

    def remove_parents_junction_nodes(self, masters):
        """ From the masters remove the ones Code which act like `Until`.

        :param masters:
        :return:
        """
        for action_set in self.is_member_of_():# self.get_masters(member=True, depend=False, join_=False, or_=False):
            for relation in ActionSet.recipe["relations"]:
                # Check first if it's a parent to other Codes
                # Firstly, check whether the link is a member type indeed
                if relation["label"].lower() != "member":
                    continue
                # if it's a receiving end of the member link and the other end (i.e. head) is Code
                if relation["child"] == action_set.anchor_code.get_id() and  ActionSet.get_action_set_position(relation["head"]):
                    # Check if it's connected via Join with itself
                    for relation_pair in action_set.relations["join"]:
                        try:
                            # Exclude if its itself
                            if relation_pair["dest"][0] == self.get_my_position()[0] and \
                                relation_pair["dest"][1] == self.get_my_position()[1]:
                                break
                        except KeyError:
                            continue
                    else:
                        # Find any other nodes that are adjacent to it with `Or` or Join
                        # From those that are to be removed
                        # Get firstly recursively the
                        func = "get_masters"
                        kwargs = {"member": False, "depend": False, "join_": True, "or_": True}
                        masters_join_or, _ = action_set.recv_masters(func, remove_parents_junction_nodes=False, remove_self=True,
                                                         remove_or=False, **kwargs)
                        # masters_, _ = action_set.recv_join_or()
                        kwargs = {"member": True, "depend": True, "join_": False, "or_": False}
                        masters__strict, _ = self.recv_masters(func, remove_parents_junction_nodes=False, remove_self=True,
                                                         remove_or=False, **kwargs)
                        for pos, action_set_ in masters_join_or.items():
                            if pos not in masters__strict:
                                masters = self.remove_parent_children(action_set_, masters)
                                masters.pop(pos, None)
                        masters = self.remove_parent_children(action_set, masters)
                        masters.pop(ActionSet.get_action_set_position(action_set), None)
                        # break
                    # break
        # Remove the children that the annotator forgot to link as Member to the Code Parent
        masters = self.remove_unobedient_children(masters)
        return masters

    def get_master_of(self, **kwargs):
        action_sets = [action_set for action_set in self.get_masters(**kwargs)]
        action_sets = set(action_sets)
        action_sets_ids = [action_set.anchor_code.get_id() for action_set in action_sets]
        for _, _, action_set in ActionSet.get_idx(ActionSet.action_sets):
            for relation in action_set.relations["depend"]:
                if relation["child"] == self.anchor_code.get_id() or relation["child"] in action_sets_ids:
                    action_sets.add(action_set)                    
        for action_set in action_sets:
            yield action_set

    def recv_masters(self, func, masters=None, paths=None, remove_parents_junction_nodes=True, remove_self=True, remove_or=True,
                     **kwargs):
        # It's not a DAG        
        if not paths:
            paths = defaultdict(list)
        if not masters:
            masters = dict()
        masters[self.get_my_position()] = self
        # Remove the adjacent Or and their dependencies if any
        to_remove = list()
        if remove_or:
            for action_set in getattr(self, func)(member=False, depend=False, join_=False, or_=True):
                # If the current node Code does have its own masters
                if list(getattr(self, func)(member=True, depend=True, join_=True, or_=False)):
                # Check if the `or` related has its own dependencies
                    for action_set_pair in getattr(action_set, func)(member=True, depend=True, join_=True, or_=False):
                        # If so remove
                        to_remove.append(ActionSet.get_action_set_position(action_set_pair))
                # if self.relations["depend"] or self.relations["member"]:
                #     continue_ = True # this was in get_masters
                to_remove.append(ActionSet.get_action_set_position(action_set))
        for action_set in getattr(self, func)(**kwargs):#  self. get_masters():
            if action_set.get_my_position() in paths[self.get_my_position()]:
                continue
            if action_set.get_my_position() in to_remove:
                continue
            else:
                # print(action_set.anchor_code.out())
                paths[self.get_my_position()].append(action_set.get_my_position())
                masters_, paths_ = action_set.recv_masters(func, masters=masters, paths=paths, remove_parents_junction_nodes=False,
                                                         remove_self=False, remove_or=False, **kwargs)
                masters = masters | masters_
                [paths[k].append(v) for k, v in paths_.items() if v not in paths[k]]
        if remove_self:
            del masters[self.get_my_position()]
        if remove_parents_junction_nodes and masters:
            masters = self.remove_parents_junction_nodes(masters)
        if to_remove:
            for k in to_remove:
                masters.pop(k, None)
        return masters, paths
        
    def recv_masters_first(self, func, masters_join_or, masters=None, deep=None, paths=None, iam_self=True, **kwargs):
        # It's not a DAG
        # print(self.anchor_code.out())
        if not paths:
            paths = defaultdict(list)
        if not masters:
            masters = dict()
            deep = 0
        masters[self.get_my_position()] = (self, deep)
        # Remove the adjacent Or and their dependencies if any
        deep += 1
        for action_set in getattr(self, func)(**kwargs):  # self. get_masters():
            if action_set.get_my_position() in paths[self.get_my_position()]:
                continue
            else:
                # print(action_set.anchor_code.out())
                paths[self.get_my_position()].append(action_set.get_my_position())
                masters_, paths_ = action_set.recv_masters_first(func, masters_join_or, masters=masters, paths=paths, deep=deep,
                                                                 iam_self=False, **kwargs)
                masters = masters | masters_
                [paths[k].append(v) for k, v in paths_.items() if v not in paths[k]]
        if iam_self:
            # Get the deepest node possible
            for k, v in dict(sorted(masters.items(), key=lambda ac: (ac[1][1]), reverse=True)).items():
                for k_, v_ in masters_join_or.items():
                    if k == k_:
                        return v[0]
        else:
            return masters, paths

    def get_dependency_chain(self):
        try:
            kwargs = {"member": False, "depend": False, "or_": False, "remove_parents_junction_nodes": False, "remove_or": False}
            master, _ = list(self.recv_masters("get_master_of", masters=None, paths=None, **kwargs))
            del kwargs['depend']
            del kwargs['or_']
            depend, _ = list(self.recv_masters("get_masters", masters=None, paths=None, **kwargs))
            cands = master | depend
            for action_set in cands.values():
                yield action_set
        except TypeError:
            raise NotImplementedError

    def is_member_of_(self):
        memberships = set()
        # Check if it has a Member relationship itself
        for relation in ActionSet.recipe["relations"]:
            if relation["label"].lower() == "member" and relation["head"] == self.anchor_code.get_id():
                try:
                    i, j = ActionSet.get_action_set_position(relation["child"])
                    memberships.add(ActionSet.action_sets[i][j])
                except TypeError:
                    raise TypeError("There should be an error with the relationship")
        else:
            # If not, check for others in its dependency chain
            for action_set in self.get_dependency_chain():
                for relation in ActionSet.recipe["relations"]:
                    if relation["head"] == action_set.anchor_code.get_id() and relation["label"].lower() == "member":
                        i, j = ActionSet.get_action_set_position(relation["child"])
                        memberships.add(ActionSet.action_sets[i][j])
                        # memberships.add(ActionSet.recipe["text"][relation["child_span"]["start"]: relation["child_span"]["end"]])
        # Find whether the ActionSet in the Membership has a Join or `Or` relationship with another ActionSet,
        # if so add them
        memberships_ = set()
        for action_set in memberships:
            for relation_type in ["or", "join"]:
                for relation in action_set.relations[relation_type]:
                    try:
                        memberships_.add(ActionSet.action_sets[relation["dest"][0]][relation["dest"][1]])
                    except KeyError:
                        continue
        memberships = memberships_ | memberships
        return memberships

    def get_action_set_children(self):
        for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
            for relation in action_set.mem_relations:
                if relation["child"] == self.anchor_code.get_id():
                    yield action_set

    def get_lu_by_id(self, id):
        for lu in self.lu:
            if lu.get_id() == id:
                return lu
        return None

    def get_entities_and_states_ids(self):
        """ Get the LUs on this ActionSet that Entity type and subclass

        :return:
        """
        for lu in self.lu:
            if isinstance(lu, ne.Entity):
                yield lu.get_id()

    def recv_lu(self, lu):
        ids = list(self.get_entities_and_states_ids())
        for relation in ActionSet.recipe["relations"]:
            if relation["head"] not in ids or relation["child"] not in ids:
                continue
            elif relation["head"] == lu.get_id():
                if self.get_lu_by_id(relation["child"]).primes_filled: #todo check condition
                    return self.get_lu_by_id(relation["child"])
                # else:
                #     return self.recv_lu(self.get_lu_by_id(relation["child"])) # todo check me
            # The below is not typically the case but may occur recursively
            elif relation["child"] == lu.get_id():
                if self.get_lu_by_id(relation["head"]).primes_filled:
                    return self.get_lu_by_id(relation["head"])

    def fill_in_entities(self, repeat=0):
        if not self.lu:
            self.entity_matched = True
        if self.entity_matched: #  and self not in or_join_stack:
            return
        # Enlist candidates from ActionSets that are before it
        cands = dict()
        # Get its Masters
        masters, _ = self.recv_masters("get_masters", masters=None, paths=None)
        # Check whether they are linked with an `Or` or a Join
        func = "get_masters"
        kwargs = {"member": False, "depend": False, "join_": True, "or_": True}
        masters_join_or, _ = self.recv_masters(func, remove_parents_junction_nodes=False, remove_self=False,
                                              remove_or=False, **kwargs)
        # masters_join_or, _ = self.recv_join_or()
        kwargs = {"member": False, "depend": False, "or_": False, "remove_parents_junction_nodes": False, "remove_or": False}
        master_of, _ = list(self.recv_masters("get_master_of", masters=None, paths=None, **kwargs))
        # Run with the order of the number of LUs they have
        if masters_join_or:  # todo check me
            priority = dict()
            for pos, action_set in masters_join_or.items():
                # Check if self is a master of action_set
                if pos in master_of:
                    continue
                func = "get_masters"
                kwargs = {"member": True, "depend": True, "join_": False, "or_": False}
                action_set_parent = action_set.recv_masters_first(func, masters_join_or, **kwargs)
                priority[action_set_parent] = len(action_set_parent.lu)
            # Rank by value of number of LUs and string hash to avoid loops
            priority = dict(sorted(priority.items(), key=lambda ac: (ac[1], repr(ac[0].__hash__ )), reverse=True))
            for action_set in priority.keys():
                if action_set.entity_matched:
                    continue
                elif action_set == self:
                    break
                else:
                    action_set.fill_in_entities()

        if masters:
            kwargs = {"member": True, "depend": False, "join_": True, "or_": True}
            orphan_untils , _ = self.recv_masters(func, remove_parents_junction_nodes=False, remove_self=False,
                                      remove_or=False, **kwargs)
            excl = [self]
            for action_set in orphan_untils.values():
                if not action_set.is_member_of_():
                    excl.append(action_set)
            # Exclude the ones that
            for action_set in masters.values():
                if action_set in masters_join_or.values() or action_set in excl:
                    continue
                # Check its depended ActionSets
                if not action_set.entity_matched:
                    action_set.fill_in_entities()
                # Get the previous masters LUs, if already Entity Matched
                for lu in action_set.lu:
                    # Do only for Entity types belong to ne.Entity and subclasses
                    try:
                        cands[lu.out()] = (lu.primes , lu.get_entity_type(), lu.get_id())
                    except AttributeError:
                        continue
        elif masters_join_or:
            for pos, action_set in masters_join_or.items():
                if action_set == self or pos in master_of:
                    continue
                # Get the masters from the Or
                log.debug("\n Self:" + self.anchor_code.out())
                log.debug("\n Masters of:" + action_set.anchor_code.out())
                masters_, _ = action_set.recv_masters("get_masters", masters=None, paths=None)
                for action_set_ in masters_.values():
                    # log.debug(action_set_.anchor_code.out())
                    # Check its depended ActionSets
                    if not action_set_.entity_matched:
                        log.debug(action_set_.anchor_code.out())
                        action_set_.fill_in_entities()
                    # Get the previous masters LUs, if already Entity Matched
                    for lu in action_set_.lu:
                        # Do only for Entity types belong to ne.Entity and subclasses
                        try:
                            cands[lu.out()] = (lu.primes, lu.get_entity_type(), lu.get_id())
                        except AttributeError:
                            continue

        for lu in self.lu:
            # Check only for Entity types belong to ne.Entity and subclasses
            try:
                if lu.primes_filled:
                    continue
            except AttributeError:
                continue
            # Check from the previous entities, if there is already
            if cands.get(lu.out(), False):
                for prime in cands[lu.out()][0]:
                    lu.add_prime(prime) # todo check me
                if lu.primes:
                    lu.primes_filled = True
                    continue
            # Check if it's similar to the actual word then assign
            if isinstance(lu, ne.Entity): # Includes Entity and State # todo check me
                # Check first the INGR, TOOL and similar COR_INGR and COR_TOOL
                if lu.get_entity_type() == "ingr":
                    for prime in self.action_set_psv["INGR"]:
                        # Updated to Partial ratio
                        if fz.fuzz.partial_ratio(prime, lu.out()) > 90:
                            lu.add_prime(prime)
                            lu.primes_filled = True
                            break
                elif  lu.get_entity_type() == "tool":
                    for prime in self.action_set_psv["TOOL"]:
                        if fz.fuzz.partial_ratio(prime, lu.out()) > 90:
                            lu.add_prime(prime)
                            lu.primes_filled = True
                            break
        # Check if it's linked to an already assigned entity
        # Rule of hierarchy
        for lu in self.lu:
            # Check only for Entity types belong to ne.Entity and subclasses
            try:
                if lu.primes_filled:
                    continue
            except AttributeError:
                continue
            higher_en = self.recv_lu(lu)
            if higher_en:
                lu.primes = higher_en.primes
                lu.primes_filled = True
        # Check for pronouns for Co-reference (pronouns etc)
        for lu in self.lu:
            # Check only for Entity types belong to ne.Entity and subclasses
            try:
                if lu.primes_filled:
                    continue
            except AttributeError:
                continue
            if not lu.primes:
                if pos_tag([lu.out()])[0][1] in ["PRP", "PRP$", "DT", "EX", "FW", "PDT", "WDT", "WP", "WP$"]:
                    if lu.get_entity_type() == "ingr":
                        for cand in cands.values():
                            if cand[1] == "ingr":
                                for prime in cand[0]:
                                    lu.add_prime(prime)
                        if lu.primes:
                            lu.primes_filled = True
                    elif lu.get_entity_type() == "tool":
                        for cand in cands.values():
                            if cand[1] == "tool":
                                for prime in cand[0]:
                                    lu.add_prime(prime)
                        if lu.primes:
                            lu.primes_filled = True
        # If the above failed have a second round of cands matching
        for lu in self.lu:
            try:
                if lu.primes_filled:
                    continue
            except AttributeError:
                continue
            # Check from the previous entities
            for cand_key in cands.keys():
                # If it's a subset and the entity type matches
                if fz.fuzz.partial_ratio(cand_key, lu.out()) > 90 and cands[cand_key][1] == lu.get_entity_type():
                    for prime in cands[cand_key][0]:
                        lu.add_prime(prime)
            if lu.primes:
                lu.primes_filled = True

        # Assert, if fail repeat
        for lu in self.lu:
            # Check only for Entity types belong to ne.Entity and subclasses
            try:
                if lu.primes_filled:
                    continue
            except AttributeError:
                continue
            if not lu.primes:
                # Give some chances to find each other
                if repeat < 10:
                    repeat += 1
                    return  self.fill_in_entities(repeat=repeat)
                else:
                    primes = set()
                    for cand in cands.values():
                        if cand[1] == lu.get_entity_type():
                            for c in cand[0]:
                                primes.add(c)
                    if primes:
                        lu.primes = primes
                        lu.primes_filled = True
                        repeat = 0
                        return self.fill_in_entities(repeat=repeat)
                    else:
                        # If everything failed
                        # Produce an error and logging
                        log.error("Entity not filled!")
                        log.error(lu.out())
                        log.error(lu.get_id())
                        # use pseudo prime
                        lu.primes_filled = True
                        raise NotImplementedError
        self.entity_matched = True

    def fill_in_modifier(self, idx):
        for relation in ActionSet.recipe['relations']:
            if relation["label"].lower() == "modifier":
                if relation['head'] == self.lu[idx].get_id():
                    # According to the Guidelines they should belong under the same ActionSet
                    try:
                        self.lu[idx].add_modifier(self.get_lu_by_id(relation['child']).out())
                    except AttributeError:
                        # It is modifying the Code as a parameter
                        if relation['child'] == self.anchor_code.get_id():
                            self.lu[idx].add_modifier(self.anchor_code.out())
                            return
                        # If failed brute force the rest of the LUs from other ActionSets
                        else:
                            for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
                                for lu in action_set.lu:
                                    if lu.get_id() == relation['child']:
                                        self.lu[idx].add_modifier(lu.out())
                                        break
                                else:
                                    if action_set.anchor_code.get_id() == relation['child']:
                                        self.lu[idx].add_modifier(action_set.anchor_code.out())
                            if not self.lu[idx].get_modifier():
                                raise NotImplementedError
                            warnings.warn("Modifier is linked cross ActionSet.")
                    except TypeError:
                        raise NotImplementedError
                    break

    def fill_in_entity_aux_links(self, idx):
        self.fill_in_modifier(idx)
        if self.lu[idx].get_modifier():
            return
        # Find the Entity that is not explicitly mentioned
        if not self.lu[idx].modifying:
            if self.lu[idx].get_label() == "sett":
                if ActionSet.annotator_added:
                    # Check if there are TOOLs that are not assigned, they are added retrospectively by the annotator
                    # Remove all the assigned
                    tools = copy.copy(self.action_set_psv["TOOL"])
                    for lu in self.lu:
                        try:
                            tools.remove(list(lu.primes)[0])
                        except (ValueError, AttributeError, IndexError):
                            pass
                    # The one left would be the one added by the annotator
                    if tools:
                        # -1 to prioritise the annotator added, which are on the last column of the spreadsheet
                        self.lu[idx].add_modifier(tools[-1])
                        return
                # Check previous ActionSets
                depend = list(self.recv_masters("get_masters", masters=None, paths=None))
                cands = dict()
                for action_set in depend[0].values():  #todo check me
                    for lu in action_set.lu:
                        try:
                            if lu.get_entity_type() == "tool":
                                cands[lu.get_id()] = lu.out()
                        except AttributeError:
                            pass
                # Get the latest tool (before the sett)
                try:
                    self.lu[idx].add_modifier(cands[min(cands)])
                except ValueError:
                    log.error("The token '" + self.lu[idx].out() + "' could not be linked with parent Entity")
                    raise NotImplementedError
            else:
                if ActionSet.annotator_added:
                    # Check if there are TOOLs that are not assigned, they are added retrospectively by the annotator
                    # Remove all the assigned
                    ingrs = copy.copy(self.action_set_psv["INGR"])
                    for lu in self.lu:
                        try:
                            ingrs.remove(list(lu.primes)[0])
                        except (ValueError, AttributeError):
                            pass
                    # The one left would be the one added by the annotator
                    if ingrs:
                        self.lu[idx].add_modifier(ingrs[0])
                        return
                # Check previous ActionSets
                depend = list(self.recv_masters("get_masters", masters=None, paths=None))
                cands = dict()
                for action_set in depend[0].values():
                    for lu in action_set.lu:
                        # `msr` can refer to INGR and TOOLs, but we give priority to ingrs unless not found
                        # Although in theory the `msr` could refer to tool, it wouldn't be on another ActionSet
                        if self.lu[idx].get_label() == "msr" and lu.get_entity_type() == "ingr":
                            cands[lu.get_id()] = lu.out()
                if not cands:  # This section is so rare I doubt it will ever be played
                    for action_set in depend[0].values():
                        for lu in action_set.lu:
                            # Although in theory the `msr` could refer to tool, it wouldn't be on another ActionSet
                            if self.lu[idx].get_label() == "msr" and lu.get_entity_type() == "tool":
                                cands[lu.get_id()] = lu.out()
                            # Get the latest tool (before the sett)
                try:
                    self.lu[idx].add_modifier(cands[min(cands)])
                except ValueError:
                    log.error("The token '" + self.lu[idx].out() + "' could not be linked with parent Entity")
                    raise NotImplementedError
            if not self.lu[idx].get_modifier:
                log.error("The token '" + self.lu[idx].out() + "' could not be linked with parent Entity")
                raise NotImplementedError

    def fill_in_entity_or_links(self, idx):
        for relation in ActionSet.recipe['relations']:
            if relation["label"].lower() == "or":
                if relation['head'] == self.lu[idx].get_id():
                    self.lu[idx].add_the_or(self.get_lu_by_id(relation['child']).out())
                elif relation['child'] == self.lu[idx].get_id():
                    self.lu[idx].add_the_or(self.get_lu_by_id(relation['head']).out())

    def fill_in_code_or_links(self):
        for relation in ActionSet.recipe['relations']:
            if relation["label"].lower() == "or":
                if relation['head'] == self.anchor_code.get_id():
                    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
                        if action_set.anchor_code.get_id() == relation['child']:
                            self.anchor_code.add_the_or(action_set.anchor_code.out())
                            break
                elif relation['child'] == self.anchor_code.get_id():
                    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
                        if action_set.anchor_code.get_id() == relation['head']:
                            self.anchor_code.add_the_or(action_set.anchor_code.out())
                            break
            elif relation["label"].lower() == "join":
                if relation['head'] == self.anchor_code.get_id():
                    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
                        if action_set.anchor_code.get_id() == relation['child']:
                            self.anchor_code.add_the_join(action_set.anchor_code.out())
                            break
                elif relation['child'] == self.anchor_code.get_id():
                    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
                        if action_set.anchor_code.get_id() == relation['head']:
                            self.anchor_code.add_the_join(action_set.anchor_code.out())
                            break
            elif relation["label"].lower() == "modifier":
                # Is Modifying an Until or an If
                if relation['head'] == self.anchor_code.get_id() and relation["child_span"]["label"].lower() in stv.code:
                    for action_set in list(itertools.chain.from_iterable(list(ActionSet.action_sets.values()))):
                        if action_set.anchor_code.get_id() == relation['child']:
                            self.anchor_code.add_modifier(action_set.anchor_code.out())
                            break

    def fill_in_links(self):
        """ Some LU's take links of modifier or Or. Here we need to fill them in.
        Here we found them and link to their LU they are referring and fill in with their primes.
        """
        for idx, lu in enumerate(self.lu):
            # print(lu.out())
            match lu:
                case x if isinstance(x, ne.EntityAux):
                    try:
                        self.fill_in_entity_aux_links(idx)
                    except NotImplementedError:
                        pass
                case x if isinstance(x, ne.State):
                    self.fill_in_modifier(idx)
                case x if isinstance(x, ne.Entity):
                    self.fill_in_entity_or_links(idx)
        # Fill in auxiliary links of the anchor_code (Code, Action) as well
        self.fill_in_code_or_links()

    def get_join_or(self):
        """ To be implemented
        """
        pass

    def add_lu(self, e):
        self.lu.append(e)

    def get_lu_out(self, token_id):
        """ Return the output for a Lexical Unit that belongs to this ActionSet

        :param token_id:
        :return:
        """
        # todo Make sure if there is an `Or` or `Join` ActionSet, to reflect their owndership there as well
        # todo check for Or/join in lu.membership
        lu = self.get_lu_by_id(token_id)
        if lu:
            # get the memberships
            mems = {self.anchor_code.out()}
            for pos in lu.memberships:
                mems.add(ActionSet.action_sets[pos[0]][pos[1]].anchor_code.out())
            mems = ", ".join(mems)
            # get the primes
            try:
                primes = lu.get_primes_()
            except AttributeError:
                pass
            except NotImplementedError:
                primes = lu.out()
            match lu:
                case x if type(x) == ne.Entity or type(x) == ne.State:
                    return "[" + \
                        lu.out() + " | " + \
                        primes + " | " + \
                        mems + " | " + \
                        lu.get_label().upper() + " | " + \
                        lu.link_out() + "]"
                case x if type(x) == ne.EntityAux:
                    return "[" + \
                        lu.out() + " | " + \
                        " | " + \
                        mems + " | " + \
                        lu.get_label().upper() + " | " + \
                        lu.link_out() + "]"
                case x if type(x) == ne.Why:
                    return "[" + \
                        lu.out() + " | " + \
                        " | " + \
                        mems + " | " + \
                        lu.get_label().upper() + " | " + \
                        "]"

    def get_anchor_code_out(self):
        """

        Note: Deal with the Join and Or
        Note: Use LM imputation to deal with missing Code
        :return:
        """
        if self.is_member_of_():
            # Make them strings
            memberships = [action_set.anchor_code.out() for action_set in self.is_member_of_()]
            member_of = ", ".join(memberships)
        else:
            member_of = self.anchor_code.out()
        deps = ""
        for dep in self.relations["depend"]:
            deps = deps + " Dependency = " + \
                   ActionSet.recipe["text"][dep["child_span"]["start"]: dep["child_span"]["end"]] + ", "
        deps = deps[:-2]
        links = self.anchor_code.link_out()
        out_relations = " | "
        if deps not in ["", " "] :
            if links:
                out_relations +=  deps + ", " + links
            else:
                out_relations += deps
        else:
            out_relations += links

        return "[" +\
            self.anchor_code.out() + " | " +\
            " | " +\
            member_of + " | " +\
            self.anchor_code.span["label"].upper() +\
            out_relations + "]"

    def update_relations(self, **kwargs):
        if kwargs.get("member"):
            self.relations["member"].extend(kwargs.get("member"))
        if kwargs.get("depend"):
            self.relations["depend"].extend(kwargs.get("depend"))
        if kwargs.get("or"):
            self.relations["or"].extend(kwargs.get("or"))
        if kwargs.get("join"):
            self.relations["join"].extend(kwargs.get("join"))

    def __init__(self, action_set_psv,anchor_code=None, **kwargs):  #  dep_relations=None, mem_relations=None,
        self.action_set_psv = action_set_psv
        self.anchor_code = anchor_code
        self.lu = list()
        self.entity_matched = False
        keys = ["member", "depend", "or", "join"]
        self.relations = {k: list() for k in keys}
        self.update_relations(**kwargs)






