import flet as ft
from hands import HandDetect  # Importa la clase HandDetect desde hands.py
from mesh import FaceMeshDetect  # Importa la clase FaceMeshDetect desde mesh.py
from pose import PoseDetect

def main(page: ft.Page):
    
    # Configurar el tema
    page.theme_mode = ft.ThemeMode.LIGHT  # Aunque sea un tema claro, los colores personalizados serán oscuros.
    page.title = "Sistema de Detección - Ciadet"
    page.bgcolor = ft.colors.BLACK87  # Fondo oscuro similar al fondo del logo.
    
    # Dropdown para seleccionar la función
    selected_function = ft.Dropdown(
        label="Selecciona la función",
        options=[
            ft.dropdown.Option("Face Mesh"),
            ft.dropdown.Option("Hand Tracking"),
            ft.dropdown.Option("Pose Tracking"),
        ],
        width=300,
        bgcolor=ft.colors.BLACK87,
        color=ft.colors.WHITE,
    )
    
    # Video stream con borde y fondo personalizado
    video_stream = ft.Image(
        src=" ",
        error_content=ft.Icon(ft.icons.PHOTO_CAMERA_FRONT, size=40, color=ft.colors.WHITE),
        fit=ft.ImageFit.CONTAIN,
        expand=True,
        border_radius=ft.border_radius.all(10),
        # border=ft.border.all(2, ft.colors.WHITE),
        # bgcolor=ft.colors.BLACK87,
    )
    
    # Contenedor para el video stream
    container_stream = ft.Container(
        content=ft.Stack(
            [
                video_stream,
            ],
            alignment=ft.alignment.center,
        ),
        padding=ft.padding.all(10),
        border_radius=ft.border_radius.all(15),
        border=ft.border.all(2, ft.colors.WHITE),
        bgcolor=ft.colors.BLACK87,
        expand=True,
    )

    # Instancias de las clases HandDetect, FaceMeshDetect, PoseDetect
    hand_detect = HandDetect(video_stream, page)
    face_mesh_detect = FaceMeshDetect(video_stream, page)
    pose_detect = PoseDetect(video_stream, page)

    def start_video_stream(e=None):
        hand_detect.play_video()  # Iniciar la transmisión de video (sin detección)

    def handle_start(e):
        if selected_function.value == "Face Mesh":
            hand_detect.stop()
            pose_detect.stop()
            face_mesh_detect.play_detect()
        
        elif selected_function.value == "Hand Tracking":
            face_mesh_detect.stop()
            pose_detect.stop()
            hand_detect.play_detect()
        
        elif selected_function.value == "Pose Tracking":
            face_mesh_detect.stop()
            hand_detect.stop()
            pose_detect.play_detect()
    
    def handle_stop(e):
        hand_detect.stop()
        face_mesh_detect.stop()
        pose_detect.stop()

    # Iniciar el streaming de video al cargar la página
    start_video_stream()

    # Botones de control estilizados
    start_button = ft.ElevatedButton(
        "Start",
        icon=ft.icons.PLAY_ARROW,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            color=ft.colors.GREEN_300,  # Usar el verde similar al punto verde del logo.
            bgcolor=ft.colors.BLACK87,  # Botón con fondo oscuro.
        ),
        on_click=handle_start,
    )

    stop_button = ft.ElevatedButton(
        "Stop",
        icon=ft.icons.STOP,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=10),
            color=ft.colors.RED_300,  # Mantener rojo para "Stop" pero con fondo oscuro.
            bgcolor=ft.colors.BLACK87,
        ),
        on_click=handle_stop,
    )

    page.add(
        ft.Column(
            controls=[
                ft.Text("Sistema de Detección - Ciadet", style=ft.TextTheme.headline_medium, color=ft.colors.WHITE),
                selected_function,
                container_stream,
                ft.Row(
                    controls=[start_button, stop_button],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=20,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            expand=True,
        )
    )

ft.app(target=main, port=8550, view=ft.WEB_BROWSER)




