@app.route('/')
def index():
    return render_template('points.html')

@app.route('/points')
def points_list():
    return render_template('points_list.html')

@app.route('/api/points')
def get_points():
    # Здесь должна быть логика получения списка точек доступа
    points = [
        {
            'id': '1',
            'name': 'AP-1',
            'band': '2.4 GHz',
            'is_online': True,
            'clients_count': 15,
            'channel': '1',
            'signal_strength': -65
        },
        {
            'id': '2',
            'name': 'AP-2',
            'band': '5 GHz',
            'is_online': True,
            'clients_count': 8,
            'channel': '36',
            'signal_strength': -70
        },
        {
            'id': '3',
            'name': 'AP-3',
            'band': '2.4 GHz',
            'is_online': False,
            'clients_count': 0,
            'channel': '6',
            'signal_strength': None
        }
    ]
    return jsonify(points) 