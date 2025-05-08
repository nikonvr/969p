import streamlit as st
import pygame
import numpy as np
import random
import sys 

# --- Paramètres Globaux ---
# Ces valeurs servent de défauts pour les widgets Streamlit
DEFAULT_NUM_DOTS = 20000
DEFAULT_NBSEGMENTS = 20
DEFAULT_NBSUBSEGMENTS = 20
DEFAULT_NBSUBSEGMENTFLOWER = 20
DEFAULT_NBSERROR666 = 100

# CORRECTION: Redéfinir les constantes globales qui ne sont pas des inputs Streamlit
flowersizeratiomin = 5
flowersizeratiomax = 20

# --- Constantes Pygame et de Conversion ---
# Utiliser des dimensions fixes pour la surface de génération
# Streamlit redimensionnera l'image pour l'affichage si nécessaire
IMG_WIDTH = 1280  
IMG_HEIGHT = 720 
# FPS n'est plus nécessaire pour le contrôle de vitesse progressif

FIG_DPI = 100 
MPL_UNIT_TO_PX_X = IMG_WIDTH
MPL_UNIT_TO_PX_Y = IMG_HEIGHT
MATPLOTLIB_Y_ASPECT_FACTOR = (16.0/9.0) / 2.0

# Couleurs Pygame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- Fonctions Utilitaires (Identiques à la version Pygame) ---
def modify_color_original_logic(rgb_0_1_tuple):
    r, g, b = rgb_0_1_tuple
    change_amount = random.uniform(0.01, 0.05)
    operation = random.choice(['add', 'sub'])
    if operation == 'add':
        r, g, b = min(1.0, r + change_amount), min(1.0, g + change_amount), min(1.0, b + change_amount)
    else:
        r, g, b = max(0.0, r - change_amount), max(0.0, g - change_amount), max(0.0, b - change_amount)
    return r, g, b

def rgb_01_to_pygame_color(rgb_0_1_tuple, alpha=255):
    r, g, b = np.clip(rgb_0_1_tuple[0],0.0,1.0), np.clip(rgb_0_1_tuple[1],0.0,1.0), np.clip(rgb_0_1_tuple[2],0.0,1.0)
    a = np.clip(int(alpha), 0, 255)
    return (int(r * 255), int(g * 255), int(b * 255), a)

def scale_coords_mpl_to_pygame(x_mpl, y_mpl):
    return int(x_mpl * MPL_UNIT_TO_PX_X), int(y_mpl * MPL_UNIT_TO_PX_Y)

def mpl_point_size_to_pygame_radius(s_mpl_points_squared):
    if s_mpl_points_squared <= 0: return 0
    radius_px = (np.sqrt(s_mpl_points_squared) * FIG_DPI / 72.0) / 2.0
    return max(1, int(radius_px))

def mpl_linewidth_to_pygame_thickness(linewidth_mpl_points):
    thickness_px = linewidth_mpl_points * FIG_DPI / 72.0
    return max(1, int(thickness_px))

# --- Fonctions de Génération de Données (Adaptées pour prendre les paramètres) ---
def generate_branch_and_flower_data_mpl(start_x_mpl, start_y_mpl, nbsubsegments, nbsubsegmentflower, flowersizeratiomin, flowersizeratiomax):
    branch_segment_data = []
    flower_creation_data = []
    x_mpl, y_mpl = start_x_mpl, start_y_mpl
    angle_rad = 0
    subsegment_counter_for_flower = 0

    for _ in range(nbsubsegments): # Utilise le paramètre nbsubsegments
        subsegment_length_mpl = np.random.uniform(0.01, 0.04)
        subsegment_width_param_mpl = np.random.uniform(0.005, 0.02)
        max_subsegment_angle_rad = np.radians(40)
        angle_rad += np.random.uniform(-max_subsegment_angle_rad, max_subsegment_angle_rad)
        prev_x_mpl, prev_y_mpl = x_mpl, y_mpl
        x_mpl += subsegment_length_mpl * np.cos(angle_rad)
        y_mpl += subsegment_length_mpl * np.sin(angle_rad)

        branch_segment_data.append({
            'x1_mpl': prev_x_mpl, 'y1_mpl': prev_y_mpl,
            'x2_mpl': x_mpl, 'y2_mpl': y_mpl,
            'mpl_linewidth_points': subsegment_width_param_mpl * 100
        })
        subsegment_counter_for_flower += 1
        if subsegment_counter_for_flower >= nbsubsegmentflower: # Utilise le paramètre nbsubsegmentflower
            flower_data = {
                'base_x_mpl': x_mpl, 'base_y_mpl': y_mpl,
                'initial_size_mpl': np.random.uniform(0.000002, 0.000004) * np.random.uniform(flowersizeratiomin, flowersizeratiomax),
                'initial_color_01': (random.randint(0, 1) * 1.0, random.randint(0, 1) * 1.0, random.randint(0, 1) * 1.0),
                'orientation': np.random.randint(-1, 2), 'shape_id': np.random.randint(1, 5)
            }
            flower_creation_data.append(flower_data)
            subsegment_counter_for_flower = 0
    return branch_segment_data, flower_creation_data

# --- Fonctions de Dessin (Adaptées pour ne plus gérer l'affichage progressif) ---

BACKGROUND_PALETTES_01 = [
    [(0.2,0.1,0.05),(0.1,0.05,0.02),(0.4,0.25,0.1),(0.5,0.3,0.15),(0.6,0.3,0.1),(0.7,0.4,0.2),(0.8,0.5,0.2),(0.85,0.6,0.3),(0.1,0.2,0.05),(0.2,0.3,0.1),(0.3,0.4,0.1),(0.4,0.5,0.2),(0.05,0.05,0.05),(0.1,0.1,0.1),(0.85,0.8,0.7),(0.9,0.85,0.75)],
    [(0.1,0.15,0.05),(0.15,0.2,0.1),(0.2,0.3,0.1),(0.3,0.45,0.15),(0.4,0.6,0.2),(0.5,0.7,0.3),(0.1,0.25,0.15),(0.2,0.35,0.2),(0.05,0.05,0.08),(0.7,0.7,0.6)]
]

def draw_background_dots_pygame(surface, num_dots): # Prend num_dots en paramètre
    x_coords_mpl = np.random.rand(num_dots)
    y_coords_mpl = np.random.rand(num_dots)
    active_palette_01 = random.choice(BACKGROUND_PALETTES_01)
    dot_params_mpl = [
        {'min_s': 50, 'max_s': 200, 'min_alpha': 0.1, 'max_alpha': 0.5, 'large_alpha_min': 0.5, 'large_alpha_max': 1.0},
        {'min_s': 5, 'max_s': 20, 'min_alpha': 0.1, 'max_alpha': 0.5, 'large_alpha_min': 0.5, 'large_alpha_max': 1.0},
        {'min_s': 1, 'max_s': 2, 'min_alpha': 0.5, 'max_alpha': 1.0, 'large_alpha_min': 0.9, 'large_alpha_max': 1.0}
    ]
    for params in dot_params_mpl:
        dot_sizes_mpl_s = np.random.uniform(params['min_s'], params['max_s'], num_dots)
        dot_alphas_01 = np.random.uniform(params['min_alpha'], params['max_alpha'], num_dots)
        if num_dots >= 100:
            largest_dot_indices = np.argpartition(dot_sizes_mpl_s, -100)[-100:]
            dot_alphas_01[largest_dot_indices] = np.random.uniform(params['large_alpha_min'], params['large_alpha_max'], len(largest_dot_indices))
        for i in range(num_dots):
            radius_px = mpl_point_size_to_pygame_radius(dot_sizes_mpl_s[i])
            if radius_px == 0: continue
            base_color_01 = random.choice(active_palette_01)
            r_01 = np.clip(base_color_01[0] + random.uniform(-0.05, 0.05), 0.0, 1.0)
            g_01 = np.clip(base_color_01[1] + random.uniform(-0.05, 0.05), 0.0, 1.0)
            b_01 = np.clip(base_color_01[2] + random.uniform(-0.05, 0.05), 0.0, 1.0)
            final_dot_color_pygame = rgb_01_to_pygame_color((r_01, g_01, b_01), alpha=dot_alphas_01[i]*255) 
            px, py = scale_coords_mpl_to_pygame(x_coords_mpl[i], y_coords_mpl[i])
            temp_surface = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, final_dot_color_pygame, (radius_px, radius_px), radius_px)
            surface.blit(temp_surface, (px - radius_px, py - radius_px))

def draw_one_branch_segment_pygame(surface, seg_data):
    """Dessine un unique segment de branche."""
    start_pos_px = scale_coords_mpl_to_pygame(seg_data['x1_mpl'], seg_data['y1_mpl'])
    end_pos_px = scale_coords_mpl_to_pygame(seg_data['x2_mpl'], seg_data['y2_mpl'])
    thickness_px = mpl_linewidth_to_pygame_thickness(seg_data['mpl_linewidth_points'])
    pygame.draw.line(surface, BLACK, start_pos_px, end_pos_px, thickness_px)

def draw_flower_pygame(drawing_surface, flower_data, flowersizeratiomin, flowersizeratiomax): # Ne prend plus screen/clock
    """Dessine une fleur complète sur drawing_surface."""
    flower_x_mpl_array = np.full(100, flower_data['base_x_mpl']) 
    flower_y_mpl_array = np.full(100, flower_data['base_y_mpl'])
    current_size_mpl = flower_data['initial_size_mpl']
    current_color_01 = flower_data['initial_color_01'] 
    orientation = flower_data['orientation']
    shape_id = flower_data['shape_id']
    t_array = np.linspace(0, 2 * np.pi, 100)

    for _ in range(50): # 50 "traces"
        current_size_mpl += np.random.uniform(0.0000002, 0.000002) * np.random.uniform(flowersizeratiomin, flowersizeratiomax)
        size_multiplier = 1.0
        if shape_id == 2: size_multiplier = 5.0
        elif shape_id == 3: size_multiplier = 2.0
        effective_size_mpl = current_size_mpl * size_multiplier
        dx_mpl, dy_mpl = np.zeros_like(t_array), np.zeros_like(t_array)
        if shape_id == 1:
            dx_mpl = 0.5*effective_size_mpl*(np.random.uniform(1,3)*np.sin(t_array) + orientation*np.random.uniform(0.0125,0.25)*np.sin(2*t_array))
            dy_mpl = MATPLOTLIB_Y_ASPECT_FACTOR*effective_size_mpl*(np.random.uniform(1,3)*np.cos(t_array) - orientation*np.random.uniform(0.0125,0.25)*np.cos(2*t_array))
        elif shape_id == 2:
            dx_mpl = 0.5*effective_size_mpl*(np.random.uniform(0.25,0.5)*np.sin(t_array) + orientation*np.random.uniform(0.25,0.5)*np.sin(2*t_array))
            dy_mpl = MATPLOTLIB_Y_ASPECT_FACTOR*effective_size_mpl*(np.random.uniform(0.25,0.5)*np.cos(t_array) - orientation*np.random.uniform(0.25,0.5)*np.cos(2*t_array))
        elif shape_id == 3:
            dx_mpl = 0.5*effective_size_mpl*(0.25*np.sin(t_array) + 0.75*np.sin(2*t_array))
            dy_mpl = MATPLOTLIB_Y_ASPECT_FACTOR*effective_size_mpl*(0.25*np.cos(t_array) - 0.75*np.cos(2*t_array))
        elif shape_id == 4:
            dx_mpl = 0.5*effective_size_mpl*(np.random.uniform(0.25,1)*np.sin(t_array) + orientation*np.random.uniform(0.5,2)*np.sin(2*t_array) + np.sin(5*t_array))
            dy_mpl = MATPLOTLIB_Y_ASPECT_FACTOR*effective_size_mpl*(np.random.uniform(0.25,1)*np.cos(t_array) - orientation*np.random.uniform(0.5,2)*np.cos(2*t_array) + np.cos(5*t_array))
        
        flower_x_mpl_array += dx_mpl
        flower_y_mpl_array += dy_mpl
        current_color_01 = modify_color_original_logic(current_color_01)
        final_trace_color_pygame = rgb_01_to_pygame_color(current_color_01) # Alpha par défaut 255
        pointlist_px = [(scale_coords_mpl_to_pygame(flower_x_mpl_array[j], flower_y_mpl_array[j])) for j in range(len(flower_x_mpl_array))]
        
        # Dessiner la trace actuelle
        if len(pointlist_px) > 1:
            try: pygame.draw.aalines(drawing_surface, final_trace_color_pygame[:3], False, pointlist_px) # Utiliser RGB pour aalines
            except Exception as e_draw: print(f"Erreur Pygame draw.aalines: {e_draw}")
        
        # PAS d'affichage progressif ici dans la version Streamlit

    # Dessiner les "extensions de feuille" (scatter) APRES toutes les traces
    size_for_scatter_mpl_scalar = current_size_mpl - np.random.uniform(0,0.00002)*np.random.uniform(flowersizeratiomin,flowersizeratiomax)
    current_color_01 = modify_color_original_logic(current_color_01) 
    final_scatter_color_pygame = rgb_01_to_pygame_color(current_color_01) # Alpha par défaut 255
    for i in range(0, len(flower_x_mpl_array), 5): 
        base_x_scatter_mpl = flower_x_mpl_array[i]
        base_y_scatter_mpl = flower_y_mpl_array[i]
        offset_x_mpl = np.random.uniform(-0.0005,0.0005)*np.random.uniform(flowersizeratiomin,flowersizeratiomax)
        offset_y_mpl = np.random.uniform(-0.0005,0.0005)*np.random.uniform(flowersizeratiomin,flowersizeratiomax)
        scatter_point_x_mpl = base_x_scatter_mpl + offset_x_mpl
        scatter_point_y_mpl = base_y_scatter_mpl + offset_y_mpl
        px_scatter, py_scatter = scale_coords_mpl_to_pygame(scatter_point_x_mpl, scatter_point_y_mpl)
        scatter_radius_px = 1 
        if scatter_radius_px > 0:
            pygame.draw.circle(drawing_surface, final_scatter_color_pygame[:3], (px_scatter, py_scatter), scatter_radius_px) # Utiliser RGB pour circle
            
    # PAS d'affichage progressif ici

def draw_error_texts_pygame(surface, nbserror666): # Prend nbserror666 en paramètre
    # Initialisation de la police ici car pygame.init() n'est pas appelé globalement
    try:
        pygame.font.init() 
        default_font = pygame.font.Font(None, 24) # Taille par défaut si la taille calculée échoue
    except Exception as e:
        st.error(f"Impossible d'initialiser le module font de Pygame: {e}")
        return # Ne pas essayer de dessiner du texte si la police échoue

    for _ in range(nbserror666):
        x_pos_mpl,y_pos_mpl = random.uniform(0,1),random.uniform(0,1)
        font_size_mpl_points = random.randint(2,48)
        font_size_px = mpl_linewidth_to_pygame_thickness(font_size_mpl_points)
        gray_level_01 = random.uniform(0.1,1)
        text_color_pygame = (int(gray_level_01*255),int(gray_level_01*255),int(gray_level_01*255))
        text_str = f"Error {random.choice('69')}{random.choice('69')}{random.choice('69')}"
        try: 
            current_font = pygame.font.Font(None, font_size_px)
        except: 
            current_font = default_font # Fallback
        try:
            text_surface = current_font.render(text_str, True, text_color_pygame)
            px,py = scale_coords_mpl_to_pygame(x_pos_mpl,y_pos_mpl)
            text_rect = text_surface.get_rect(center=(px,py))
            surface.blit(text_surface, text_rect)
        except Exception as e_render:
             # Peut échouer si la taille de police est trop grande ou invalide
             print(f"Erreur rendu texte: {e_render} pour taille {font_size_px}")
             pass 
    
    # Quitter le module font après usage
    # pygame.font.quit() # Peut causer des problèmes si appelé avant la fin de l'app Streamlit

# --- Fonction Principale de Génération pour Streamlit ---
# @st.cache_data # Optionnel: Mettre en cache si la génération est très longue et les params ne changent pas souvent
def generate_final_image(num_dots, nbsegments, nbsubsegments, nbsubsegmentflower, nbserror666, flowersizeratiomin, flowersizeratiomax):
    """Génère l'image complète sur une surface Pygame et retourne un tableau NumPy."""
    
    # Initialiser Pygame (nécessaire pour utiliser les modules comme surfarray)
    pygame.init() 

    # Créer la surface de dessin
    drawing_surface = pygame.Surface((IMG_WIDTH, IMG_HEIGHT))
    drawing_surface.fill(BLACK)

    print("Génération de l'image...")
    
    # Dessiner l'arrière-plan
    draw_background_dots_pygame(drawing_surface, num_dots)
    print("Fond terminé.")

    # Dessiner les branches et les fleurs
    for i in range(nbsegments):
        start_x_mpl, start_y_mpl = np.random.uniform(0,0.9), np.random.uniform(0,0.9)
        branch_segments, flower_instructions = generate_branch_and_flower_data_mpl(
            start_x_mpl, start_y_mpl, nbsubsegments, nbsubsegmentflower, flowersizeratiomin, flowersizeratiomax
        )
        
        # Dessiner les segments de la branche
        for seg_data in branch_segments:
            draw_one_branch_segment_pygame(drawing_surface, seg_data)
            
        # Dessiner les fleurs de la branche
        for flower_data in flower_instructions:
            draw_flower_pygame(drawing_surface, flower_data, flowersizeratiomin, flowersizeratiomax)
            
        if (i+1) % (nbsegments // 5 or 1) == 0: 
            print(f"Branche {i+1}/{nbsegments} générée.")

    print("Segments et fleurs terminés.")
    
    # Dessiner les textes "Error"
    draw_error_texts_pygame(drawing_surface, nbserror666)
    print("Textes 'Error' terminés.")
    
    print("Conversion de l'image pour Streamlit...")
    # Convertir la surface Pygame en tableau NumPy pour Streamlit
    # Utiliser pygame.surfarray.array3d() est souvent plus direct
    img_array = pygame.surfarray.array3d(drawing_surface)
    # Pygame donne (width, height, channels), Streamlit attend (height, width, channels)
    img_array = img_array.swapaxes(0, 1) 
    
    print("Génération terminée.")
    
    # Quitter Pygame après usage
    pygame.quit() 
    
    return img_array

# --- Interface Streamlit ---
st.set_page_config(layout="wide") # Utiliser toute la largeur
st.title("Générateur d'Art Floral - Streamlit")
st.markdown("Ajustez les paramètres dans la barre latérale et cliquez sur 'Générer l'Image'.")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres de Génération")
num_dots_input = st.sidebar.number_input("Nombre de points (fond)", min_value=1000, max_value=50000, value=DEFAULT_NUM_DOTS, step=1000, help="Densité des points en arrière-plan.")
nbsegments_input = st.sidebar.number_input("Nombre de branches", min_value=1, max_value=100, value=DEFAULT_NBSEGMENTS, step=1, help="Nombre de branches principales.")
nbsubsegments_input = st.sidebar.number_input("Segments par branche", min_value=5, max_value=50, value=DEFAULT_NBSUBSEGMENTS, step=1, help="Longueur et complexité de chaque branche.")
nbsubsegmentflower_input = st.sidebar.number_input("Fréquence des fleurs (moins = plus)", min_value=1, max_value=50, value=DEFAULT_NBSUBSEGMENTFLOWER, step=1, help="Nombre de segments avant une nouvelle fleur.")
nbserror666_input = st.sidebar.number_input("Nombre de textes 'Error'", min_value=0, max_value=500, value=DEFAULT_NBSERROR666, step=10, help="Nombre de messages 'Error' aléatoires.")

# Bouton pour lancer la génération
if st.sidebar.button("Générer l'Image"):
    
    # Afficher un message pendant la génération
    with st.spinner("Génération de l'image en cours... Cela peut prendre un moment."):
        # Appel de la fonction de génération avec les paramètres de l'interface
        # Utilisation des constantes globales redéfinies au début du script
        final_image_array = generate_final_image(
            num_dots=num_dots_input,
            nbsegments=nbsegments_input,
            nbsubsegments=nbsubsegments_input,
            nbsubsegmentflower=nbsubsegmentflower_input,
            nbserror666=nbserror666_input,
            flowersizeratiomin=flowersizeratiomin, 
            flowersizeratiomax=flowersizeratiomax  
        )
    
    # Afficher l'image générée
    st.image(final_image_array, caption="Image Générée", use_column_width=True)
    
    st.success("Génération terminée !")

else:
    # Message initial ou si le bouton n'est pas cliqué
    st.info("Cliquez sur 'Générer l'Image' dans la barre latérale pour commencer.")

# Ajouter des informations ou des instructions supplémentaires si nécessaire
st.sidebar.markdown("---")
st.sidebar.info("Application créée à partir d'un script Pygame/Matplotlib.")

