"""
This module contains function to parse a GLM into a dictionary that can then be
modified and then exported to a modified glm

    parse(inputStr, filePath=True):
        Main function to parse in glm
    _tokenize_glm(inputStr, filePath=True):
        Helper function to parse glm
    _parse_token_list(tokenList):
        Helper function to parse glm
    sorted_write(inTree):
        Main function to write out glm
    _dict_to_string(inDict):
        Helper function to write out glm
    _gather_key_values(inDict, keyToAvoid):
        Helper function to write out glm


Adopted August 30, 2018 by Brandon Thayer (brandon.thayer@pnnl.gov)
Modified March 28, 2017 by Jacob Hansen (jacob.hansen@pnnl.gov)
Created October 27, 2014 by Ebony Mayhorn (ebony.mayhorn@pnnl.gov)

Copyright (c) 2018 Battelle Memorial Institute.  The Government retains a paid-up nonexclusive,
irrevocable worldwide license to reproduce, prepare derivative works, perform publicly and display
publicly by or for the Government, including the right to distribute to other Government
contractors.
"""

import re
import warnings
from functools import reduce


def parse(input_str, file_path=True):
    """
    Parse a GLM into an omf.feeder tree. This is so we can walk the tree,
    change things in bulk, etc.

    Input can be a file path or GLM string.
    """

    tokens = _tokenize_glm(input_str, file_path)
    return _parse_token_list(tokens)


def _tokenize_glm(input_str, file_path=True):
    """ Turn a GLM file/string into a linked list of tokens.

    E.g. turn a string like this:
    clock {clockey valley;};
    object house {name myhouse; object ZIPload {inductance bigind; power
        newpower;}; size 234sqft;};

    Into a Python list like this:
    ['clock','{','clockey','valley','}','object','house','{','name','myhouse',
        ';','object','ZIPload','{','inductance','bigind',';','power',
        'newpower','}','size','234sqft','}']
    """

    if file_path:
        with open(input_str, 'r') as glmFile:
            data = glmFile.read()

    else:
        data = input_str
    # Get rid of http for stylesheets because we don't need it and it conflicts
    # with comment syntax.
    data = re.sub(r'http:\/\/', '', data)
    # Strip comments.
    data = re.sub(r'\/\/.*\n', '', data)
    # Also strip non-single whitespace because it's only for humans:
    data = data.replace('\r', '').replace('\t', ' ')
    # Tokenize around semicolons, braces and whitespace.
    tokenized = re.split(r'(;|\}|\{|\s)', data)
    # Get rid of whitespace strings.
    basic_list = [x for x in tokenized if x != '' and x != ' ']
    return basic_list


def _parse_token_list(token_list):
    """
    Given a list of tokens from a GLM, parse those into a tree data structure.

    """

    def current_leaf_add(key_f, value, tree_f, guid_stack_f):
        # Helper function to add to the current leaf we're visiting.
        current = tree_f
        for x in guid_stack_f:
            current = current[x]
        current[key_f] = value

    def list_to_string(list_in):
        # Helper function to turn a list of strings into one string with some
        # decent formatting.
        if len(list_in) == 0:
            return ''
        else:
            return reduce(lambda x, y: str(x) + ' ' + str(y), list_in[1:-1])

    # Tree variables.
    tree = {}
    guid = 0
    guid_stack = []

    # reverse the token list as pop() is way more efficient than pop(0)
    token_list = list(reversed(token_list))

    while token_list:
        # Pop, then keep going until we have a full token (i.e. 'object house',
        # not just 'object')
        full_token = []
        while full_token == [] or full_token[-1] not in ['{', ';', '}', '\n',
                                                         'shape']:
            full_token.append(token_list.pop())
        # Work with what we've collected.
        if full_token[0] == '#set':
            if full_token[-1] == ';':
                tree[guid] = {'omftype': full_token[0],
                              'argument': list_to_string(full_token)}
            else:
                tree[guid] = {'#set': list_to_string(full_token)}
            guid += 1
        elif full_token[0] == '#include':
            if full_token[-1] == ';':
                tree[guid] = {'omftype': full_token[0],
                              'argument': list_to_string(full_token)}
            else:
                tree[guid] = {'#include': list_to_string(full_token)}
            guid += 1
        elif full_token[0] == 'shape':
            while full_token[-1] not in ['\n']:
                full_token.append(token_list.pop())
            full_token[-2] = ''
            current_leaf_add(full_token[0], list_to_string(full_token[0:-1]),
                             tree, guid_stack)
            guid += 1
        elif full_token[-1] == '\n' or full_token[-1] == ';':
            # Special case when we have zero-attribute items (like #include,
            # #set, module).
            if guid_stack == [] and full_token != ['\n'] and \
                    full_token != [';']:

                tree[guid] = {'omftype': full_token[0],
                              'argument': list_to_string(full_token)}
                guid += 1
            # We process if it isn't the empty token (';')
            elif len(full_token) > 1:
                current_leaf_add(full_token[0], list_to_string(full_token),
                                 tree, guid_stack)
        elif full_token[-1] == '}':
            if len(full_token) > 1:
                current_leaf_add(full_token[0], list_to_string(full_token),
                                 tree, guid_stack)
            guid_stack.pop()
        elif full_token[0] == 'schedule':
            # Special code for those ugly schedule objects:
            if full_token[0] == 'schedule':
                while full_token[-1] not in ['}']:
                    full_token.append(token_list.pop())
                tree[guid] = {'object': 'schedule', 'name': full_token[1],
                              'cron': ' '.join(full_token[3:-2])}
                guid += 1
        elif full_token[-1] == '{':
            current_leaf_add(guid, {}, tree, guid_stack)
            guid_stack.append(guid)
            guid += 1
            # Wrapping this current_leaf_add is defensive coding so we don't
            # crash on malformed glm files.
            if len(full_token) > 1:
                # Do we have a clock/object or else an embedded configuration
                # object?
                if len(full_token) < 4:
                    current_leaf_add(full_token[0], full_token[-2], tree,
                                     guid_stack)
                else:
                    current_leaf_add('omfEmbeddedConfigObject',
                                     full_token[0] + ' ' +
                                     list_to_string(full_token), tree,
                                     guid_stack)

    # this section will catch old glm format and translate it. Not in the most
    # robust way but should work for now
    objects_to_delete = []
    for key in list(tree.keys()):
        if 'object' in list(tree[key].keys()):
            # if no name is present and the object name is the old syntax we
            # need to be creative and pull the object name and use it
            if 'name' not in list(tree[key].keys()) and \
                    tree[key]['object'].find(':') >= 0:

                tree[key]['name'] = tree[key]['object'].replace(':', '_')

            # strip the old syntax from the object name
            tree[key]['object'] = tree[key]['object'].split(':')[0]

            # for the remaining sytax we will replace ':' with '_'
            for line in tree[key]:
                tree[key][line] = tree[key][line].replace(':', '_')

            # deleting all recorders from the files
            if tree[key]['object'] == 'recorder' or \
                    tree[key]['object'] == 'group_recorder' or \
                    tree[key]['object'] == 'collector':

                objects_to_delete.append(key)

            # if we are working with fuses let's set the mean replace time to 1
            # hour if not specified. Then we aviod a warning!
            if tree[key][
                'object'] == 'fuse' and 'mean_replacement_time' not in list(
                    tree[key].keys()):
                tree[key]['mean_replacement_time'] = 3600.0

            # FNCS is not able to handle names that include "-" so we will
            # replace that with "_"
            if 'name' in list(tree[key].keys()):
                tree[key]['name'] = tree[key]['name'].replace('-', '_')
            if 'parent' in list(tree[key].keys()):
                tree[key]['parent'] = tree[key]['parent'].replace('-', '_')
            if 'from' in list(tree[key].keys()):
                tree[key]['from'] = tree[key]['from'].replace('-', '_')
            if 'to' in list(tree[key].keys()):
                tree[key]['to'] = tree[key]['to'].replace('-', '_')

    # deleting all recorders from the files
    for keys in objects_to_delete:
        del tree[keys]

    return tree


def sorted_write(in_tree):
    """
    Write out a GLM from a tree, and order all tree objects by their key.

    Sometimes Gridlab breaks if you rearrange a GLM.
    """

    sorted_keys = sorted(list(in_tree.keys()), key=int)
    output = ''
    try:
        for key in sorted_keys:
            output += _dict_to_string(in_tree[key]) + '\n'
    except ValueError:
        raise Exception
    return output


def _dict_to_string(in_dict):
    """
    Helper function: given a single dict representing a GLM object, concatenate
    it into a string.
    """

    # Handle the different types of dictionaries that are leafs of the tree
    # root:
    if 'omftype' in in_dict:
        return in_dict['omftype'] + ' ' + in_dict['argument'] + ';'
    elif 'module' in in_dict:
        return ('module ' + in_dict['module'] + ' {\n'
                + _gather_key_values(in_dict, 'module') + '}\n')
    elif 'clock' in in_dict:
        # return 'clock {\n' + gatherKeyValues(in_dict, 'clock') + '};\n'
        # This object has known property order issues writing it out explicitly
        clock_string = 'clock {\n'
        if 'timezone' in in_dict:
            clock_string = clock_string + '\ttimezone ' + in_dict[
                'timezone'] + ';\n'
        if 'starttime' in in_dict:
            clock_string = clock_string + '\tstarttime ' + in_dict[
                'starttime'] + ';\n'
        if 'stoptime' in in_dict:
            clock_string = clock_string + '\tstoptime ' + in_dict[
                'stoptime'] + ';\n'
        clock_string = clock_string + '}\n'
        return clock_string
    elif 'object' in in_dict and in_dict['object'] == 'schedule':
        return 'schedule ' + in_dict['name'] + ' {\n' + in_dict[
            'cron'] + '\n};\n'
    elif 'object' in in_dict:
        return ('object ' + in_dict['object'] + ' {\n'
                + _gather_key_values(in_dict, 'object') + '};\n')
    elif 'omfEmbeddedConfigObject' in in_dict:
        return in_dict['omfEmbeddedConfigObject'] + ' {\n' + \
               _gather_key_values(in_dict, 'omfEmbeddedConfigObject') + '};\n'
    elif '#include' in in_dict:
        return '#include ' + in_dict['#include']
    elif '#define' in in_dict:
        return '#define ' + in_dict['#define'] + '\n'
    elif '#set' in in_dict:
        return '#set ' + in_dict['#set']
    elif 'class' in in_dict:
        prop = ''
        # this section will ensure we can get around the fact that you can't
        # have to key's with the same name!
        if 'variable_types' in list(
                in_dict.keys()) and 'variable_names' in list(
                in_dict.keys()) and len(in_dict['variable_types']) == len(
                in_dict['variable_names']):
            prop += 'class ' + in_dict['class'] + ' {\n'
            for x in range(len(in_dict['variable_types'])):
                prop += '\t' + in_dict['variable_types'][x] + ' ' + \
                        in_dict['variable_names'][x] + ';\n'

            prop += '}\n'
        else:
            prop += 'class ' + in_dict['class'] + ' {\n' + _gather_key_values(
                in_dict, 'class') + '}\n'

        return prop


def _gather_key_values(in_dict, key_to_avoid):
    """
    Helper function: put key/value pairs for objects into the format GLD needs.
    """

    other_key_values = ''
    for key in in_dict:
        if type(key) is int:
            # WARNING: RECURSION HERE
            other_key_values += _dict_to_string(in_dict[key])
        elif key != key_to_avoid:
            if key == 'comment':
                other_key_values += (in_dict[key] + '\n')
            elif key == 'name' or key == 'parent':
                if len(in_dict[key]) <= 62:
                    other_key_values += (
                                '\t' + key + ' ' + str(in_dict[key]) + ';\n')
                else:
                    warnings.warn(
                        ("{:s} argument is longer that 64 characters. "
                         + " Truncating {:s}.").format(key, in_dict[key]),
                        RuntimeWarning)
                    other_key_values += ('\t' + key + ' '
                                         + str(in_dict[key])[0:62]
                                         + '; // truncated from {:s}\n'.format(
                                            in_dict[key]))
            else:
                other_key_values += ('\t' + key + ' ' + str(in_dict[key]) + ';\n')
    return other_key_values


def _test():
    import time
    start = time.time()
    # cProfile.run('re.compile("foo|bar")')
    feeder_location = r'C:\Users\thay838\git_repos\gridappsd-pyvvo\pyvvo\tests\models\ieee8500_base.glm'
    feeder_dictionary = parse(feeder_location)
    # print(feeder_dictionary)
    feeder_str = sorted_write(feeder_dictionary)
    glm_file = open('/Users/hans464/Desktop/new_feeder.glm', 'w')
    glm_file.write(feeder_str)
    glm_file.close()
    end = time.time()
    print('successfully completed in {:0.1f} seconds'.format(end - start))


if __name__ == '__main__':
    _test()
