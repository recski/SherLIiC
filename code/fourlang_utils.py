import re

class Utils:

    def d_clean(self, string):
        s = string
        for c in '\\=@-,\'".!:;':
            s = s.replace(c, '_')
        s = s.replace('$', '_dollars')
        s = s.replace('%', '_percent')
        if s == '#':
            s = '_number'
        keywords = ("graph", "node", "strict", "edge")
        if re.match('^[0-9]', s) or s in keywords:
            s = "X" + s
        return s

    def get_edges(self, graph=None):
        lines = []
        # lines.append('\tordering=out;')
        # sorting everything to make the process deterministic

        edge_lines = []
        for u, v, edata in graph.edges(data=True):
            if 'color' in edata:
                d_node1 = self.d_clean(u)
                d_node2 = self.d_clean(v)

                lines.append((self.d_clean(d_node1).split("_")[0], self.d_clean(d_node2).split("_")[0], edata['color']))
        return lines



    def to_dot(self, graph=None):
        lines = [u'digraph finite_state_machine {', '\tdpi=100;']
        # lines.append('\tordering=out;')
        # sorting everything to make the process deterministic
        self.d_clean("asd")
        node_lines = []
        for node, n_data in graph.nodes(data=True):
            d_node = self.d_clean(node)
            printname = self.d_clean('_'.join(d_node.split('_')[:-1]))
            if 'expanded' in n_data and not n_data['expanded']:
                node_line = u'\t{0} [shape = circle, label = "{1}", \
                        style="filled"];'.format(
                                d_node, printname).replace('-', '_')
            else:
                node_line = u'\t{0} [shape = circle, label = "{1}"];'.format(
                        d_node, printname).replace('-', '_')
                node_lines.append(node_line)
        lines += sorted(node_lines)

        edge_lines = []
        for u, v, edata in graph.edges(data=True):
            if 'color' in edata:
                d_node1 = self.d_clean(u)
                d_node2 = self.d_clean(v)
                edge_lines.append(
                        u'\t{0} -> {1} [ label = "{2}" ];'.format(
                            self.d_clean(d_node1), self.d_clean(d_node2),
                            edata['color']))

        lines += sorted(edge_lines)
        lines.append('}')
        return u'\n'.join(lines)



    def asim_jac(self, seq1, seq2):
        set1, set2 = map(set, (seq1, seq2))
        intersection = set1 & set2
        if not intersection:
            return 0
        else:
            sim = float(len(intersection)) / len(set2)
            return sim
