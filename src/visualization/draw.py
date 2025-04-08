from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='Logical Graph of Appointments and Institutions')

# Person nodes
dot.node('P1', 'Philippe Jamet')
dot.node('P2', 'Pascal Ray')
dot.node('P3', 'David Delafosse')
dot.node('P4', 'Jacques Fayolle')

# Institution nodes
dot.node('MSE', 'Mines Saint-Étienne')
dot.node('IMT', 'Institut Mines-Télécom')

# Date nodes
dot.node('D1', '2008-09-01')
dot.node('D2', '2014-07-14')
dot.node('D3', '2014-07-15')
dot.node('D4', '2019-09-02')
dot.node('D5', '2021-11-30')
dot.node('D6', '2021-12-01')
dot.node('D7', '2022-04-30')
dot.node('D8', '2022-05-01')

# Appointment nodes (each represents a director appointment)
dot.node('A1', 'Appointment 1\n(Director)')
dot.node('A2', 'Appointment 2\n(Director)')
dot.node('A3', 'Appointment 3\n(Director)')
dot.node('A4', 'Appointment 4\n(Director)')
dot.node('A5', 'Appointment 5\n(Director)')

# Appointment 1: Philippe Jamet at Mines Saint-Étienne
dot.edge('A1', 'P1', label='person')
dot.edge('A1', 'MSE', label='institution')
dot.edge('A1', 'D1', label='startDate')
dot.edge('A1', 'D2', label='endDate')

# Appointment 2: Philippe Jamet at Institut Mines-Télécom
dot.edge('A2', 'P1', label='person')
dot.edge('A2', 'IMT', label='institution')
dot.edge('A2', 'D3', label='startDate')
dot.edge('A2', 'D4', label='endDate')

# Appointment 3: Pascal Ray at Mines Saint-Étienne
dot.edge('A3', 'P2', label='person')
dot.edge('A3', 'MSE', label='institution')
dot.edge('A3', 'D3', label='startDate')
dot.edge('A3', 'D5', label='endDate')

# Appointment 4: David Delafosse at Mines Saint-Étienne
dot.edge('A4', 'P3', label='person')
dot.edge('A4', 'MSE', label='institution')
dot.edge('A4', 'D6', label='startDate')
dot.edge('A4', 'D7', label='endDate')

# Appointment 5: Jacques Fayolle at Mines Saint-Étienne (ongoing)
dot.edge('A5', 'P4', label='person')
dot.edge('A5', 'MSE', label='institution')
dot.edge('A5', 'D8', label='startDate')

# Institutional relationship: Mines Saint-Étienne is part of Institut Mines-Télécom
dot.edge('MSE', 'IMT', label='isPartOf')

# Render and display the graph (this will create a file named 'logical_graph.pdf')
dot.render('logical_graph', view=True)
