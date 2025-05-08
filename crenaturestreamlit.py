import streamlit as st
import pygame
import numpy as np
import random
import time
import io

# --- Paramètres Globaux (modifiés pour la nouvelle logique) ---
flowersizeratiomin = 0.005
flowersizeratiomax = 0.02

num_dots = 10000 # Reduced for faster background in Streamlit context
nbsegments = 10 # Number of flower-stem systems
nbsubsegments = 15 # Segments per stem
nbserror666 = 50 # Number of error texts

# --- Constantes Pygame ---
# FPS is effectively controlled by Streamlit's refresh rate + time.sleep()
STREAMLIT_FPS = 30 # Target FPS for Streamlit refresh
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- Variables globales pour dimensions et échelle (seront définies dans main) ---
SCREEN_WIDTH_REF = 0
SCREEN_HEIGHT_REF = 0
REFERENCE_DIMENSION = 0
NORM_SCALING_FACTOR_X = 1.0
NORM_SCALING_FACTOR_Y = 1.0

# --- Fonctions Utilitaires ---
def modify_color_original_logic(rgb_0_1_tuple):
    r, g, b = rgb_0_1_tuple
    change_amount = random.uniform(0.01, 0.05)
    operation = random.choice(['add', 'sub'])
    if operation == 'add':
        r = min(1.0, r + change_amount)
        g = min(1.0, g + change_amount)
        b = min(1.0, b + change_amount)
    else:
        r = max(0.0, r - change_amount)
        g = max(0.0, g - change_amount)
        b = max(0.0, b - change_amount)
    return r, g, b

def rgb_01_to_pygame_color(rgb_0_1_tuple, alpha=255):
    r = np.clip(rgb_0_1_tuple[0], 0.0, 1.0)
    g = np.clip(rgb_0_1_tuple[1], 0.0, 1.0)
    b = np.clip(rgb_0_1_tuple[2], 0.0, 1.0)
    a = np.clip(int(alpha), 0, 255)
    return (int(r * 255), int(g * 255), int(b * 255), a)

# --- Fonctions de Conversion et d'Échelle ---
def update_screen_constants(width, height):
    global SCREEN_WIDTH_REF, SCREEN_HEIGHT_REF, REFERENCE_DIMENSION
    global NORM_SCALING_FACTOR_X, NORM_SCALING_FACTOR_Y

    SCREEN_WIDTH_REF = width
    SCREEN_HEIGHT_REF = height
    if SCREEN_WIDTH_REF == 0 or SCREEN_HEIGHT_REF == 0:
        REFERENCE_DIMENSION = 1
    else:
        REFERENCE_DIMENSION = min(SCREEN_WIDTH_REF, SCREEN_HEIGHT_REF)

    if SCREEN_WIDTH_REF > 0:
        NORM_SCALING_FACTOR_X = REFERENCE_DIMENSION / SCREEN_WIDTH_REF
    else:
        NORM_SCALING_FACTOR_X = 1.0

    if SCREEN_HEIGHT_REF > 0:
        NORM_SCALING_FACTOR_Y = REFERENCE_DIMENSION / SCREEN_HEIGHT_REF
    else:
        NORM_SCALING_FACTOR_Y = 1.0

def scale_coords_norm_to_pygame(x_norm, y_norm):
    return int(x_norm * SCREEN_WIDTH_REF), int(y_norm * SCREEN_HEIGHT_REF)

def scale_dimension_relative_to_pygame(relative_dimension):
    if REFERENCE_DIMENSION == 0: return 1
    return max(1, int(relative_dimension * REFERENCE_DIMENSION))

# --- Génération de Données pour la Tige ---
def generate_branch_segments_from_point_norm(start_x_norm, start_y_norm, initial_angle_rad, num_segments_to_generate):
    branch_segment_data = []
    x_norm, y_norm = start_x_norm, start_y_norm
    angle_rad = initial_angle_rad

    for _ in range(num_segments_to_generate):
        subsegment_length_relative = np.random.uniform(0.005, 0.02)
        subsegment_thickness_relative = np.random.uniform(0.001, 0.005)
        max_subsegment_angle_rad = np.radians(45)
        angle_rad += np.random.uniform(-max_subsegment_angle_rad, max_subsegment_angle_rad)
        prev_x_norm, prev_y_norm = x_norm, y_norm
        delta_x_norm = subsegment_length_relative * np.cos(angle_rad) * NORM_SCALING_FACTOR_X
        delta_y_norm = subsegment_length_relative * np.sin(angle_rad) * NORM_SCALING_FACTOR_Y
        x_norm += delta_x_norm
        y_norm += delta_y_norm
        branch_segment_data.append({
            'x1_norm': prev_x_norm, 'y1_norm': prev_y_norm,
            'x2_norm': x_norm, 'y2_norm': y_norm,
            'relative_thickness': subsegment_thickness_relative
        })
    return branch_segment_data

# --- Fonctions de Dessin (Pygame) ---
BACKGROUND_PALETTES_01 = [
    [(0.2,0.1,0.05),(0.1,0.05,0.02),(0.4,0.25,0.1),(0.5,0.3,0.15),(0.6,0.3,0.1),(0.7,0.4,0.2),(0.8,0.5,0.2),(0.85,0.6,0.3),(0.1,0.2,0.05),(0.2,0.3,0.1),(0.3,0.4,0.1),(0.4,0.5,0.2),(0.05,0.05,0.05),(0.1,0.1,0.1),(0.85,0.8,0.7),(0.9,0.85,0.75)],
    [(0.1,0.15,0.05),(0.15,0.2,0.1),(0.2,0.3,0.1),(0.3,0.45,0.15),(0.4,0.6,0.2),(0.5,0.7,0.3),(0.1,0.25,0.15),(0.2,0.35,0.2),(0.05,0.05,0.08),(0.7,0.7,0.6)]
]

def draw_background_dots_pygame(surface):
    x_coords_norm = np.random.rand(num_dots)
    y_coords_norm = np.random.rand(num_dots)
    active_palette_01 = random.choice(BACKGROUND_PALETTES_01)
    dot_params_relative = [
        {'min_radius_rel': 0.002, 'max_radius_rel': 0.01, 'min_alpha': 0.1, 'max_alpha': 0.5, 'large_alpha_min': 0.5, 'large_alpha_max': 1.0},
        {'min_radius_rel': 0.0005, 'max_radius_rel': 0.002, 'min_alpha': 0.1, 'max_alpha': 0.5, 'large_alpha_min': 0.5, 'large_alpha_max': 1.0},
        {'min_radius_rel': 0.0001, 'max_radius_rel': 0.0005, 'min_alpha': 0.5, 'max_alpha': 1.0, 'large_alpha_min': 0.9, 'large_alpha_max': 1.0}
    ]
    for params in dot_params_relative:
        dot_radii_relative = np.random.uniform(params['min_radius_rel'], params['max_radius_rel'], num_dots)
        dot_alphas_01 = np.random.uniform(params['min_alpha'], params['max_alpha'], num_dots)
        if num_dots >= 100:
            num_largest_dots = min(100, num_dots)
            if num_largest_dots > 0:
                largest_dot_indices = np.argpartition(dot_radii_relative, -num_largest_dots)[-num_largest_dots:]
                dot_alphas_01[largest_dot_indices] = np.random.uniform(params['large_alpha_min'], params['large_alpha_max'], len(largest_dot_indices))

        for i in range(num_dots):
            radius_px = scale_dimension_relative_to_pygame(dot_radii_relative[i])
            if radius_px == 0: continue
            base_color_01 = random.choice(active_palette_01)
            r_01 = np.clip(base_color_01[0] + random.uniform(-0.05, 0.05), 0.0, 1.0)
            g_01 = np.clip(base_color_01[1] + random.uniform(-0.05, 0.05), 0.0, 1.0)
            b_01 = np.clip(base_color_01[2] + random.uniform(-0.05, 0.05), 0.0, 1.0)
            final_dot_color_pygame = rgb_01_to_pygame_color((r_01, g_01, b_01), alpha=dot_alphas_01[i]*255)
            px, py = scale_coords_norm_to_pygame(x_coords_norm[i], y_coords_norm[i])
            temp_surface = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surface, final_dot_color_pygame, (radius_px, radius_px), radius_px)
            surface.blit(temp_surface, (px - radius_px, py - radius_px))

def draw_one_branch_segment_pygame(surface, seg_data):
    start_pos_px = scale_coords_norm_to_pygame(seg_data['x1_norm'], seg_data['y1_norm'])
    end_pos_px = scale_coords_norm_to_pygame(seg_data['x2_norm'], seg_data['y2_norm'])
    thickness_px = scale_dimension_relative_to_pygame(seg_data['relative_thickness'])
    branch_color_variation = random.uniform(0.8, 1.0)
    branch_base_gray = int(50 * branch_color_variation)
    branch_color = (branch_base_gray, branch_base_gray, branch_base_gray)
    try:
        pygame.draw.line(surface, branch_color, start_pos_px, end_pos_px, thickness_px)
    except Exception:
        if thickness_px > 0 :
            pygame.draw.line(surface, branch_color, start_pos_px, end_pos_px, max(1,thickness_px))


def draw_flower_growth_one_step_pygame(drawing_surface, flower_params_evo):
    flower_x_norm_array = flower_params_evo['flower_x_norm_array_evo']
    flower_y_norm_array = flower_params_evo['flower_y_norm_array_evo']
    
    flower_params_evo['current_size_relative_evo'] += np.random.uniform(0.0001, 0.0005) * np.random.uniform(flowersizeratiomin, flowersizeratiomax)
    
    current_size_relative = flower_params_evo['current_size_relative_evo']
    orientation = flower_params_evo['orientation']
    shape_id = flower_params_evo['shape_id']
    t_array = flower_params_evo['t_array']

    n_petals = flower_params_evo.get('n_petals')
    m_waves = flower_params_evo.get('m_waves')
    n_lobes_cardioid = flower_params_evo.get('n_lobes_cardioid')
    amplitude_cardioid_mod = flower_params_evo.get('amplitude_cardioid_mod')
    n_points_star = flower_params_evo.get('n_points_star')
    amplitude_star_main = flower_params_evo.get('amplitude_star_main')
    amplitude_star_mod = flower_params_evo.get('amplitude_star_mod')
    freq_star_mod_factor = flower_params_evo.get('freq_star_mod_factor')
    n_main_lobes_gear = flower_params_evo.get('n_main_lobes_gear')
    n_sub_lobes_gear = flower_params_evo.get('n_sub_lobes_gear')
    amp_main_gear = flower_params_evo.get('amp_main_gear')
    amp_sub_gear = flower_params_evo.get('amp_sub_gear')

    size_multiplier = 1.0
    if shape_id == 2: size_multiplier = 5.0
    elif shape_id == 3: size_multiplier = 2.0
    elif shape_id == 5: size_multiplier = 1.5
    elif shape_id == 6: size_multiplier = 1.2
    elif shape_id == 7: size_multiplier = 2.0
    elif shape_id == 8: size_multiplier = 1.3
    elif shape_id == 9: size_multiplier = 1.4
    effective_size_relative = current_size_relative * size_multiplier
    
    dx_shape_unit, dy_shape_unit = np.zeros_like(t_array), np.zeros_like(t_array)

    if shape_id == 1:
        dx_shape_unit = 0.5*(np.random.uniform(1,3)*np.sin(t_array) + orientation*np.random.uniform(0.0125,0.25)*np.sin(2*t_array))
        dy_shape_unit = 0.5*(np.random.uniform(1,3)*np.cos(t_array) - orientation*np.random.uniform(0.0125,0.25)*np.cos(2*t_array))
    elif shape_id == 2:
        dx_shape_unit = 0.5*(np.random.uniform(0.25,0.5)*np.sin(t_array) + orientation*np.random.uniform(0.25,0.5)*np.sin(2*t_array))
        dy_shape_unit = 0.5*(np.random.uniform(0.25,0.5)*np.cos(t_array) - orientation*np.random.uniform(0.25,0.5)*np.cos(2*t_array))
    elif shape_id == 3:
        dx_shape_unit = 0.5*(0.25*np.sin(t_array) + 0.75*np.sin(2*t_array))
        dy_shape_unit = 0.5*(0.25*np.cos(t_array) - 0.75*np.cos(2*t_array))
    elif shape_id == 4:
        dx_shape_unit = 0.5*(np.random.uniform(0.25,1)*np.sin(t_array) + orientation*np.random.uniform(0.5,2)*np.sin(2*t_array) + np.sin(5*t_array))
        dy_shape_unit = 0.5*(np.random.uniform(0.25,1)*np.cos(t_array) - orientation*np.random.uniform(0.5,2)*np.cos(2*t_array) + np.cos(5*t_array))
    elif shape_id == 5:
        k = n_petals / random.uniform(1.0, 2.0)
        rhodonea_r_unit = np.cos(k * t_array)
        dx_shape_unit = rhodonea_r_unit * np.cos(t_array)
        dy_shape_unit = rhodonea_r_unit * np.sin(t_array)
    elif shape_id == 6:
        petal_modulation = 0.4 * np.sin(m_waves * t_array + orientation * np.pi/4) + \
                           0.2 * np.cos((m_waves/2) * t_array) * orientation
        radius_t_unit = (1 + petal_modulation + np.random.uniform(-0.1, 0.1))
        dx_shape_unit = radius_t_unit * np.cos(t_array)
        dy_shape_unit = radius_t_unit * np.sin(t_array)
    elif shape_id == 7:
        cardioid_base_unit = 0.5 * (1 - np.cos(t_array + orientation * np.pi/2))
        modulation = 1 + amplitude_cardioid_mod * np.sin(n_lobes_cardioid * t_array)
        radius_t_unit = cardioid_base_unit * modulation
        dx_shape_unit = radius_t_unit * np.cos(t_array)
        dy_shape_unit = radius_t_unit * np.sin(t_array)
    elif shape_id == 8:
        main_shape_unit = amplitude_star_main * np.cos(n_points_star * t_array)
        modulation_shape_unit = amplitude_star_mod * np.sin(n_points_star * freq_star_mod_factor * t_array + orientation * np.pi/3)
        radius_t_unit = (1 + main_shape_unit + modulation_shape_unit + np.random.uniform(-0.05, 0.05))
        dx_shape_unit = radius_t_unit * np.cos(t_array)
        dy_shape_unit = radius_t_unit * np.sin(t_array)
    elif shape_id == 9:
        main_lobes_unit = amp_main_gear * np.cos(n_main_lobes_gear * t_array)
        sub_lobes_unit = amp_sub_gear * np.sin(n_sub_lobes_gear * t_array + orientation * t_array * 0.2)
        radius_t_unit = (0.8 + main_lobes_unit + sub_lobes_unit + np.random.uniform(-0.05, 0.05))
        dx_shape_unit = radius_t_unit * np.cos(t_array)
        dy_shape_unit = radius_t_unit * np.sin(t_array)

    dx_offset_norm = effective_size_relative * dx_shape_unit * NORM_SCALING_FACTOR_X
    dy_offset_norm = effective_size_relative * dy_shape_unit * NORM_SCALING_FACTOR_Y
    
    flower_params_evo['flower_x_norm_array_evo'] += dx_offset_norm
    flower_params_evo['flower_y_norm_array_evo'] += dy_offset_norm
    
    flower_params_evo['current_color_01_evo'] = modify_color_original_logic(flower_params_evo['current_color_01_evo'])
    final_trace_color_pygame = rgb_01_to_pygame_color(flower_params_evo['current_color_01_evo'])
    
    pointlist_px = [(scale_coords_norm_to_pygame(flower_params_evo['flower_x_norm_array_evo'][j], flower_params_evo['flower_y_norm_array_evo'][j])) for j in range(len(flower_x_norm_array))]
    
    if len(pointlist_px) > 1:
        try:
            pygame.draw.aalines(drawing_surface, final_trace_color_pygame[:3], False, pointlist_px)
        except Exception: # Minimal error handling for brevity
            pass


def draw_flower_scatter_points_pygame(drawing_surface, flower_params_evo):
    flower_x_norm_array = flower_params_evo['flower_x_norm_array_evo']
    flower_y_norm_array = flower_params_evo['flower_y_norm_array_evo']
    current_size_relative = flower_params_evo['current_size_relative_evo'] # Use the final size

    size_for_scatter_relative = current_size_relative - np.random.uniform(0,0.0005) * np.random.uniform(flowersizeratiomin,flowersizeratiomax)
    
    # Modify color one last time for scatter points
    flower_params_evo['current_color_01_evo'] = modify_color_original_logic(flower_params_evo['current_color_01_evo'])
    final_scatter_color_pygame = rgb_01_to_pygame_color(flower_params_evo['current_color_01_evo'])
    
    for i in range(0, len(flower_x_norm_array), 5):
        base_x_scatter_norm = flower_x_norm_array[i]
        base_y_scatter_norm = flower_y_norm_array[i]
        
        offset_magnitude_relative = size_for_scatter_relative * np.random.uniform(0.05, 0.2)
        offset_x_norm = offset_magnitude_relative * random.choice([-1,1]) * np.random.rand() * NORM_SCALING_FACTOR_X
        offset_y_norm = offset_magnitude_relative * random.choice([-1,1]) * np.random.rand() * NORM_SCALING_FACTOR_Y
        
        scatter_point_x_norm = base_x_scatter_norm + offset_x_norm
        scatter_point_y_norm = base_y_scatter_norm + offset_y_norm
        
        px_scatter, py_scatter = scale_coords_norm_to_pygame(scatter_point_x_norm, scatter_point_y_norm)
        scatter_radius_px = 1 
        
        if scatter_radius_px > 0:
            pygame.draw.circle(drawing_surface, final_scatter_color_pygame[:3], (px_scatter, py_scatter), scatter_radius_px)


def draw_error_texts_pygame(surface, num_to_draw_this_frame=1):
    default_font_size_px = scale_dimension_relative_to_pygame(0.02)
    default_font = None # Will be initialized if needed
    # This function is called multiple times, drawing a few texts per frame
    # The main Streamlit loop will control how many are drawn in total via nbserror666

    try: # pygame.font might not be initialized on first call if not careful
        if pygame.font.get_init():
             default_font = pygame.font.Font(None, default_font_size_px)
    except Exception: # pygame.error: font not initialized
        pygame.font.init() # Try to initialize it
        default_font = pygame.font.Font(None, default_font_size_px)


    for _ in range(num_to_draw_this_frame):
        x_pos_norm,y_pos_norm = random.uniform(0,1),random.uniform(0,1)
        font_size_relative = random.uniform(0.005, 0.04)
        font_size_px = scale_dimension_relative_to_pygame(font_size_relative)
        gray_level_01 = random.uniform(0.1,1)
        text_color_pygame = (int(gray_level_01*255),int(gray_level_01*255),int(gray_level_01*255))
        text_str = f"Error {random.choice('69')}{random.choice('69')}{random.choice('69')}"
        
        current_font = default_font
        try:
            if font_size_px > 0: # Ensure font size is valid
                 current_font = pygame.font.Font(None, font_size_px)
        except: # Fallback to default if specific size fails
            if default_font is None and pygame.font.get_init(): # Ensure default_font initialized
                default_font = pygame.font.Font(None, default_font_size_px if default_font_size_px > 0 else 10)
            current_font = default_font
        
        if current_font:
            text_surface = current_font.render(text_str, True, text_color_pygame)
            px,py = scale_coords_norm_to_pygame(x_pos_norm,y_pos_norm)
            text_rect = text_surface.get_rect(center=(px,py))
            surface.blit(text_surface, text_rect)

def pygame_surface_to_st_image(surface):
    img_array = pygame.surfarray.array3d(surface)
    img_array = np.transpose(img_array, (1, 0, 2)) # Pygame (w,h,c) to Numpy (h,w,c) for st.image
    return img_array


def main_streamlit():
    st.set_page_config(layout="wide", page_title="Art Génératif Streamlit")
    st.title("Art Génératif Floral Absurde - Streamlit (Option 2)")
    
    STREAMLIT_CANVAS_WIDTH = 800
    STREAMLIT_CANVAS_HEIGHT = 600

    placeholder = st.empty()
    
    # Controls
    cols = st.columns([1,1,1,2])
    if cols[0].button("Démarrer/Redémarrer", key="start_restart"):
        st.session_state.clear() # Full reset
        st.rerun()

    if 'stop_flag' not in st.session_state:
        st.session_state.stop_flag = False

    if cols[1].button("Arrêter", key="stop_anim", disabled=st.session_state.get('app_stage', 'INIT') == 'DONE'):
        st.session_state.stop_flag = True
        st.info("L'animation s'arrêtera à la prochaine étape majeure.")
    
    # Initialisation de Pygame et de la surface de dessin si nécessaire
    if 'pygame_initialized' not in st.session_state:
        pygame.init() # Initialize all Pygame modules
        pygame.font.init() # Explicitly initialize font module
        st.session_state.pygame_initialized = True
        update_screen_constants(STREAMLIT_CANVAS_WIDTH, STREAMLIT_CANVAS_HEIGHT)
        st.session_state.drawing_surface = pygame.Surface((STREAMLIT_CANVAS_WIDTH, STREAMLIT_CANVAS_HEIGHT))
        st.session_state.drawing_surface.fill(BLACK)
        st.session_state.app_stage = "INIT" 
        st.session_state.current_main_segment_idx = 0 # For nbsegments (flower-stem systems)
        st.session_state.flower_growth_step_idx = 0
        st.session_state.current_branch_subsegment_idx = 0
        st.session_state.error_drawing_count = 0


    surface = st.session_state.drawing_surface
    
    if st.session_state.stop_flag and st.session_state.app_stage not in ["DONE", "INIT"]:
        st.warning("Arrêt en cours...")
        # To prevent immediate jump to DONE, allow current drawing phase to complete partially or fully
        # For simplicity, we'll let it finish the current major step (flower or branch sequence) then stop.
        # A more graceful stop would be more complex.
        # Here, we just set to DONE to break the loop effectively.
        # st.session_state.app_stage = "DONE" # Or a specific "STOPPED" state


    # Machine d'état pour l'animation
    current_stage = st.session_state.app_stage

    if current_stage == "INIT":
        surface.fill(BLACK) # Clear surface on restart
        draw_background_dots_pygame(surface)
        st.session_state.app_stage = "INIT_FLOWER_SYSTEM"
        st.rerun()

    elif current_stage == "INIT_FLOWER_SYSTEM":
        if st.session_state.current_main_segment_idx < nbsegments and not st.session_state.stop_flag:
            # Initialize data for a new flower
            flower_base_x_norm = np.random.uniform(0.1, 0.9)
            flower_base_y_norm = np.random.uniform(0.1, 0.9)
            
            initial_size_rel = np.random.uniform(0.01, 0.03) * np.random.uniform(flowersizeratiomin, flowersizeratiomax)
            initial_color_01_val = (random.choice([0.0, 1.0]), random.choice([0.0, 1.0]), random.choice([0.0, 1.0]))

            st.session_state.current_flower_params = {
                'base_x_norm': flower_base_x_norm,
                'base_y_norm': flower_base_y_norm,
                'initial_size_relative': initial_size_rel,
                'initial_color_01': initial_color_01_val,
                'orientation': np.random.randint(-1, 2),
                'shape_id': np.random.randint(1, 10),
                't_array': np.linspace(0, 2 * np.pi, 100),
                # Evolving state
                'current_size_relative_evo': initial_size_rel,
                'current_color_01_evo': initial_color_01_val,
                'flower_x_norm_array_evo': np.full(100, flower_base_x_norm),
                'flower_y_norm_array_evo': np.full(100, flower_base_y_norm),
                # Store shape-specific random parameters once per flower
                'n_petals': random.choice([3, 5, 7, 4, 6]),
                'm_waves': random.choice([4, 6, 8, 5, 7]),
                'n_lobes_cardioid': random.choice([5,6,7,8]),
                'amplitude_cardioid_mod': random.uniform(0.1, 0.4),
                'n_points_star': random.choice([5,7,9,6,8]),
                'amplitude_star_main': random.uniform(0.4, 0.8),
                'amplitude_star_mod': random.uniform(0.1, 0.3),
                'freq_star_mod_factor': random.uniform(1.2, 2.5),
                'n_main_lobes_gear': random.choice([6,8,10,12]),
                'n_sub_lobes_gear': random.choice([12,16,20,24]),
                'amp_main_gear': random.uniform(0.15, 0.3),
                'amp_sub_gear': random.uniform(0.05, 0.15),
            }
            st.session_state.flower_growth_step_idx = 0
            st.session_state.app_stage = "DRAWING_FLOWER_GROWTH"
            st.rerun()
        else:
            st.session_state.app_stage = "INIT_ERRORS"
            st.rerun()
            
    elif current_stage == "DRAWING_FLOWER_GROWTH":
        if st.session_state.flower_growth_step_idx < 50 and not st.session_state.stop_flag: # 50 steps for flower growth
            draw_flower_growth_one_step_pygame(surface, st.session_state.current_flower_params)
            st.session_state.flower_growth_step_idx += 1
            # No rerun, fall through to display then rerun later
        else:
            st.session_state.app_stage = "DRAWING_FLOWER_SCATTER"
            st.rerun()

    elif current_stage == "DRAWING_FLOWER_SCATTER":
        if not st.session_state.stop_flag:
            draw_flower_scatter_points_pygame(surface, st.session_state.current_flower_params)
        st.session_state.app_stage = "INIT_BRANCH"
        st.rerun()

    elif current_stage == "INIT_BRANCH":
        if not st.session_state.stop_flag:
            flower_params = st.session_state.current_flower_params
            initial_branch_angle_rad = np.random.uniform(0, 2 * np.pi)
            st.session_state.current_branch_segments_data = generate_branch_segments_from_point_norm(
                flower_params['base_x_norm'], flower_params['base_y_norm'],
                initial_branch_angle_rad, nbsubsegments
            )
            st.session_state.current_branch_subsegment_idx = 0
        st.session_state.app_stage = "DRAWING_BRANCH"
        st.rerun()

    elif current_stage == "DRAWING_BRANCH":
        if st.session_state.current_branch_subsegment_idx < nbsubsegments and not st.session_state.stop_flag:
            segment_data = st.session_state.current_branch_segments_data[st.session_state.current_branch_subsegment_idx]
            draw_one_branch_segment_pygame(surface, segment_data)
            st.session_state.current_branch_subsegment_idx += 1
            # No rerun, fall through
        else:
            st.session_state.current_main_segment_idx += 1
            st.session_state.app_stage = "INIT_FLOWER_SYSTEM" # Loop back for next flower or move to errors
            st.rerun()
            
    elif current_stage == "INIT_ERRORS":
        st.session_state.error_drawing_count = 0
        st.session_state.app_stage = "DRAWING_ERRORS"
        st.rerun()

    elif current_stage == "DRAWING_ERRORS":
        if st.session_state.error_drawing_count < nbserror666 and not st.session_state.stop_flag:
            # Draw a few errors per frame for effect
            draw_error_texts_pygame(surface, num_to_draw_this_frame=2)
            st.session_state.error_drawing_count += 2 
            # No rerun, fall through
        else:
            st.session_state.app_stage = "DONE"
            st.rerun()

    elif current_stage == "DONE":
        st.success("Génération terminée!")
        # Final display handled below, no more reruns from here in the logic.
    
    # Display the current surface
    img_array = pygame_surface_to_st_image(surface)
    placeholder.image(img_array, use_column_width=True)

    if cols[2].button("Sauvegarder PNG", key="save_png"):
        img_byte_arr = io.BytesIO()
        pygame.image.save(surface, img_byte_arr, "PNG")
        img_byte_arr.seek(0)
        cols[3].download_button(
            label="Télécharger art.png",
            data=img_byte_arr,
            file_name="art_generatif_streamlit.png",
            mime="image/png"
        )

    # Rerun logic for animation steps
    if st.session_state.app_stage not in ["DONE", "INIT"] and not st.session_state.stop_flag :
        time.sleep(1 / STREAMLIT_FPS)
        st.rerun()
    elif st.session_state.stop_flag and st.session_state.app_stage not in ["DONE", "INIT"]:
        st.session_state.app_stage = "DONE" # Force done if stop_flag is true and not already done
        st.rerun()


if __name__ == '__main__':
    main_streamlit()
