import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def generate_report():
    doc = Document()
    
    # --- Title ---
    title_run = doc.add_heading(level=0)
    title_run.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_text = title_run.add_run('Multi-Robot Warehouse Navigation & Task Allocation')
    title_text.bold = True
    
    doc.add_paragraph() # Spacing
    
    # --- Table for Students ---
    table = doc.add_table(rows=5, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].paragraphs[0].add_run('Student Name').bold = True
    hdr_cells[1].paragraphs[0].add_run('Roll No.').bold = True
    hdr_cells[2].paragraphs[0].add_run('Department').bold = True
    
    for i in range(1, 5):
        row_cells = table.rows[i].cells
        row_cells[0].text = f'Student Name {i}'
        row_cells[1].text = f'Roll No. {i}'
        row_cells[2].text = f'Department {i}'
        
    doc.add_paragraph() # Spacing
    
    # --- 1. Problem Formulation ---
    doc.add_heading('1 Problem Formulation', level=1)
    
    p1 = doc.add_paragraph()
    p1.add_run('Real-world problem: ').bold = True
    p1.add_run('Warehouse logistics, optimal path planning without collisions, and efficient task assignment.')
    
    p2 = doc.add_paragraph()
    p2.add_run('Characteristics: ').bold = True
    p2.add_run('The environment is modeled as discrete grids, highly dynamic due to multiple agents operating concurrently, and constrained by time windows and spatial availability.')
    
    # --- 2. Method and Justification ---
    doc.add_heading('2 Method and Justification', level=1)
    
    doc.add_heading('Why This Method is Appropriate', level=2)
    p3 = doc.add_paragraph()
    p3.add_run('Space-Time A*: ').bold = True
    p3.add_run('Allows agents to avoid dynamic obstacles effectively by adding the time dimension.\n')
    p3.add_run('Prioritized Planning: ').bold = True
    p3.add_run('Drastically reduces the dimensionality and complexity compared to joint pathfinding, ensuring fast performance in dense warehouse environments.\n')
    p3.add_run('Greedy/Hungarian Algorithm: ').bold = True
    p3.add_run('Optimizes task assignments globally, minimizing travel costs.')

    doc.add_heading('Algorithm Design and Methodology', level=2)
    doc.add_paragraph('We implemented a modular pipeline that first clusters and allocates tasks using optimal assignment (Hungarian), and subsequently resolves collision-free space-time trajectories based on robot priority.')
    
    p4 = doc.add_paragraph()
    p4.add_run('Algorithm 1 (Space-Time A* with Prioritized Planning):').bold = True
    
    algo_text = (
        "1. Initialization: Sort all robots based on task priority or distance.\n"
        "2. Reservation Table = Empty Set\n"
        "3. For each robot in sorted list:\n"
        "4.     Path = Space_Time_A_Star(start, goal, Reservation Table)\n"
        "5.     If Path exists:\n"
        "6.         Reserve space-time coordinates of Path in Reservation Table\n"
        "7.     Else:\n"
        "8.         Return Planning Failure\n"
        "9. Return Multi-Agent Paths"
    )
    algo_p = doc.add_paragraph(algo_text)
    for run in algo_p.runs:
        run.font.name = 'Courier New'
        
    # --- 3. Implementation Details ---
    doc.add_heading('3 Implementation Details', level=1)
    
    doc.add_heading('System Design', level=2)
    doc.add_paragraph('main.py: Orchestrates task assignment and calls path planners.', style='List Bullet')
    doc.add_paragraph('simulation.py: Executes trajectories and visualizes robot movements.', style='List Bullet')
    doc.add_paragraph('astar.py: Core pathfinding logic incorporating time-based reservation lookups.', style='List Bullet')
    
    doc.add_heading('GitHub Repository', level=2)
    doc.add_paragraph('Source link: [Insert GitHub Repository Link Here]')
    
    # --- 4. Results and Performance Analysis ---
    doc.add_heading('4 Results and Performance Analysis', level=1)
    
    doc.add_heading('Experimental Setup', level=2)
    doc.add_paragraph('Validation was performed on a 22x24 grid with 4 robots resolving 9 tasks. Trajectories and metrics were visually tracked using a Matplotlib-based viz renderer.')
    
    doc.add_heading('Quantitative Results', level=2)
    res_p = doc.add_paragraph()
    res_p.add_run('Total steps: ').bold = True
    res_p.add_run('61\n')
    res_p.add_run('Total Distance: ').bold = True
    res_p.add_run('192\n')
    res_p.add_run('Avg task makespan: ').bold = True
    res_p.add_run('29.2\n')
    res_p.add_run('Throughput: ').bold = True
    res_p.add_run('14.75')
    
    output_path = os.path.join(os.path.dirname(__file__), 'Course_Project_Report.docx')
    doc.save(output_path)
    print(f"Report generated successfully at: {output_path}")

if __name__ == "__main__":
    generate_report()
