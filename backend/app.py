from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import shutil
from torchvision.utils import save_image
from sqlalchemy import inspect
from werkzeug.utils import secure_filename
import re
from sqlalchemy import text

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# 配置静态文件目录
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'results'), exist_ok=True)

# 添加根路由
@app.route('/')
def index():
    return app.send_static_file('index.html')

# 数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:21013067@localhost/icafusion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True  # 启用SQL语句日志
db = SQLAlchemy(app)

def get_current_time():
    """获取当前东八区时间"""
    return datetime.now(timezone(timedelta(hours=8)))

# 用户模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=get_current_time)
    is_admin = db.Column(db.Boolean, default=False)

# 检测记录模型
class DetectionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    visible_image_path = db.Column(db.String(255), nullable=False)
    infrared_image_path = db.Column(db.String(255), nullable=False)
    result_image_path = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone(timedelta(hours=8))))
    detection_details = db.Column(db.JSON, nullable=True)  # 添加检测详情字段
    image_sequence = db.Column(db.String(50), nullable=True)  # 修改为字符串类型

# 创建数据库表
with app.app_context():
    try:
        # 使用 inspect 检查表是否存在
        inspector = inspect(db.engine)
        if not inspector.has_table('user'):
            db.create_all()
            logger.info("数据库表创建成功")
        else:
            # 检查detection_record表中的image_sequence字段类型
            columns = inspector.get_columns('detection_record')
            for column in columns:
                if column['name'] == 'image_sequence' and str(column['type']) != 'VARCHAR(50)':
                    # 如果字段类型不是VARCHAR(50)，则修改表结构
                    with db.engine.connect() as conn:
                        conn.execute(text('ALTER TABLE detection_record MODIFY COLUMN image_sequence VARCHAR(50)'))
                        conn.commit()
                    logger.info("已修改image_sequence字段类型为VARCHAR(50)")
            logger.info("数据库表已存在")
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        raise

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('weights/ICAFusion_FLIR.pt', map_location=device)
model.eval()

def save_uploaded_file(file, prefix):
    """保存上传的文件"""
    if not file.filename:
        raise ValueError('没有选择文件')
        
    # 生成唯一的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = secure_filename(f'{prefix}_{timestamp}_{file.filename}')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 保存文件
    file.save(filepath)
    logger.info(f"文件已保存: {filepath}")
    
    return os.path.join('uploads', filename)

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        logger.debug(f"收到的注册数据: {data}")
        
        if not data:
            return jsonify({'error': '没有收到数据'}), 400
            
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400
            
        logger.debug(f"尝试注册用户: {username}")
        
        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            logger.warning(f"用户名已存在: {username}")
            return jsonify({'error': '用户名已存在'}), 400
        
        # 创建新用户
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        
        try:
            db.session.add(user)
            db.session.commit()
            logger.info(f"用户注册成功: {username}")
            return jsonify({'message': '注册成功'})
        except Exception as e:
            db.session.rollback()
            logger.error(f"数据库操作失败: {str(e)}")
            return jsonify({'error': '注册失败，请稍后重试'}), 500
            
    except Exception as e:
        logger.error(f"注册过程发生错误: {str(e)}")
        return jsonify({'error': '服务器内部错误'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        logger.debug(f"收到的登录数据: {data}")
        
        if not data:
            return jsonify({'error': '没有收到数据'}), 400
            
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400
            
        logger.debug(f"尝试登录用户: {username}")
        
        # 查找用户
        user = User.query.filter_by(username=username).first()
        if not user:
            logger.warning(f"用户不存在: {username}")
            return jsonify({'error': '用户名或密码错误'}), 401
            
        # 验证密码
        is_valid = check_password_hash(user.password_hash, password)
        logger.debug(f"密码验证结果: {is_valid}")
        
        if is_valid:
            logger.info(f"用户登录成功: {username}")
            return jsonify({
                'message': '登录成功',
                'user_id': user.id,
                'username': user.username,
                'is_admin': user.is_admin
            })
        else:
            logger.warning(f"密码错误: {username}")
            return jsonify({'error': '用户名或密码错误'}), 401
            
    except Exception as e:
        logger.error(f"登录过程发生错误: {str(e)}")
        return jsonify({'error': '服务器内部错误'}), 500

def extract_image_sequence(filename):
    """从文件名中提取图片序号，保持为字符串格式以保留前导零"""
    try:
        # 使用正则表达式匹配数字
        match = re.search(r'\d+', filename)
        if match:
            return match.group()  # 直接返回匹配到的字符串，不转换为整数
        return None
    except Exception as e:
        logger.error(f"提取图片序号失败: {str(e)}")
        return None

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        if 'visible_image' not in request.files or 'infrared_image' not in request.files:
            return jsonify({'error': '请上传两张图片'}), 400
            
        visible_image = request.files['visible_image']
        infrared_image = request.files['infrared_image']
        user_id = request.form.get('user_id')
        
        if not user_id:
            return jsonify({'error': '用户未登录'}), 401
            
        # 提取图片序号
        visible_sequence = extract_image_sequence(visible_image.filename)
        infrared_sequence = extract_image_sequence(infrared_image.filename)
        
        # 校验图片序号是否一致
        if not visible_sequence or not infrared_sequence:
            return jsonify({'error': '无法从文件名中提取图片序号'}), 400
            
        if visible_sequence != infrared_sequence:
            return jsonify({'error': '可见光图片和红外图片的序号不一致'}), 400
            
        logger.info(f"提取的图片序号: 可见光={visible_sequence}, 红外={infrared_sequence}")
        
        # 保存上传的图片
        visible_path = save_uploaded_file(visible_image, 'visible')
        infrared_path = save_uploaded_file(infrared_image, 'infrared')
        
        # 生成结果图片路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'result_{timestamp}.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results', result_filename)
        
        # 执行检测
        result_image, detection_details = perform_detection(visible_path, infrared_path, result_path)
        
        # 保存检测记录
        record = DetectionRecord(
            user_id=user_id,
            visible_image_path=visible_path,
            infrared_image_path=infrared_path,
            result_image_path=os.path.join('uploads', 'results', result_filename),
            detection_details=detection_details,
            image_sequence=visible_sequence  # 使用提取的图片序号
        )
        db.session.add(record)
        db.session.commit()
        
        return jsonify({
            'result_image': os.path.join('uploads', 'results', result_filename),
            'detection_details': detection_details
        })
        
    except Exception as e:
        logger.error(f"检测失败: {str(e)}")
        return jsonify({'error': '检测失败'}), 500

def perform_detection(visible_path, infrared_path, result_path):
    try:
        # 获取完整的文件路径
        visible_full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(visible_path))
        infrared_full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(infrared_path))
        
        # 加载图片
        visible_img = cv2.imread(visible_full_path)
        infrared_img = cv2.imread(infrared_full_path)
        
        if visible_img is None or infrared_img is None:
            raise ValueError('无法读取图片文件')
        
        # 获取原始图片尺寸
        h, w = visible_img.shape[:2]
        
        # 计算调整后的尺寸，保持宽高比
        img_size = 640
        scale = min(img_size/w, img_size/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整图片大小，保持比例
        visible_img_resized = cv2.resize(visible_img, (new_w, new_h))
        infrared_img_resized = cv2.resize(infrared_img, (new_w, new_h))
        
        # 创建画布
        canvas = np.zeros((new_h, new_w*2, 3), dtype=np.uint8)
        
        # 将调整后的图片放在画布上
        canvas[:, :new_w] = visible_img_resized
        canvas[:, new_w:] = infrared_img_resized
        
        # 转换为模型输入格式
        visible_img = cv2.cvtColor(visible_img_resized, cv2.COLOR_BGR2RGB)
        infrared_img = cv2.cvtColor(infrared_img_resized, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor并移动到正确的设备
        visible_img = torch.from_numpy(visible_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        infrared_img = torch.from_numpy(infrared_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # 将输入数据移动到与模型相同的设备
        visible_img = visible_img.to(device)
        infrared_img = infrared_img.to(device)
        
        # 执行推理
        with torch.no_grad():
            pred = model(visible_img, infrared_img, augment=False)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)
        
        # 处理检测结果
        visible_np = visible_img[0].cpu().permute(1, 2, 0).numpy()
        infrared_np = infrared_img[0].cpu().permute(1, 2, 0).numpy()
        
        # 创建结果图片（与画布相同大小）
        result_img = canvas.copy()
        
        # 确保图片数组是连续的
        result_img = np.ascontiguousarray(result_img)
        visible_part = np.ascontiguousarray(result_img[:, :new_w])
        infrared_part = np.ascontiguousarray(result_img[:, new_w:])
        
        # 存储检测结果详细信息
        detection_details = []
        
        for i, det in enumerate(pred):
            if len(det):
                # 将边界框坐标从模型输出格式转换为图片坐标
                det[:, :4] = scale_coords((new_h, new_w), det[:, :4], visible_img.shape[2:]).round()
                
                # 在可见光图片上绘制检测框
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, visible_part, label=label)
                    # 添加检测结果详细信息
                    detection_details.append({
                        'class': model.names[int(cls)],
                        'confidence': float(conf),
                        'box': [int(x) for x in xyxy]
                    })
                
                # 在红外图片上绘制检测框
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, infrared_part, label=label)
        
        # 将绘制好的部分放回结果图片
        result_img[:, :new_w] = visible_part
        result_img[:, new_w:] = infrared_part
        
        # 保存结果图片
        cv2.imwrite(result_path, result_img)
        logger.info(f"检测结果已保存到: {result_path}")
        
        return os.path.join('uploads', 'results', os.path.basename(result_path)), detection_details
        
    except Exception as e:
        logger.error(f"检测过程发生错误: {str(e)}")
        raise

# 添加静态文件路由
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        logger.debug(f"请求访问文件: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"访问文件失败: {filename}, 错误: {str(e)}")
        return jsonify({'error': '文件访问失败'}), 404

@app.route('/api/detection_records', methods=['GET'])
def get_detection_records():
    try:
        user_id = request.args.get('user_id')
        sort_order = request.args.get('sort_order', 'desc')
        
        # 构建查询
        query = DetectionRecord.query.filter_by(user_id=user_id)
        
        # 根据排序参数设置排序
        if sort_order == 'desc':
            query = query.order_by(DetectionRecord.created_at.desc())
        else:
            query = query.order_by(DetectionRecord.created_at.asc())
            
        records = query.all()
        
        return jsonify([{
            'id': record.id,
            'user_id': record.user_id,
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'visible_image_path': record.visible_image_path,
            'infrared_image_path': record.infrared_image_path,
            'result_image_path': record.result_image_path,
            'detection_details': record.detection_details,
            'image_sequence': record.image_sequence  # 添加图片序号
        } for record in records])
    except Exception as e:
        logger.error(f"获取检测记录失败: {str(e)}")
        return jsonify({'error': '获取检测记录失败'}), 500

@app.route('/api/update_password', methods=['POST'])
def update_password():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '没有收到数据'}), 400
            
        old_password = data.get('old_password')
        new_password = data.get('new_password')
        user_id = request.args.get('user_id')
        
        if not all([old_password, new_password, user_id]):
            return jsonify({'error': '参数不完整'}), 400
            
        # 查找用户
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        # 验证旧密码
        if not check_password_hash(user.password_hash, old_password):
            return jsonify({'error': '原密码错误'}), 401
            
        # 更新密码
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        return jsonify({'message': '密码修改成功'})
        
    except Exception as e:
        logger.error(f"修改密码失败: {str(e)}")
        db.session.rollback()
        return jsonify({'error': '修改密码失败'}), 500

@app.route('/api/admin/users', methods=['GET'])
def get_all_users():
    try:
        # 获取所有用户信息
        users = User.query.all()
        return jsonify([{
            'id': user.id,
            'username': user.username,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'is_admin': user.is_admin
        } for user in users])
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        return jsonify({'error': '获取用户列表失败'}), 500

@app.route('/api/admin/all_records', methods=['GET'])
def get_all_records():
    try:
        # 获取筛选参数
        user_id = request.args.get('user_id')
        
        # 构建查询
        query = DetectionRecord.query
        
        # 如果指定了用户ID，添加用户筛选条件
        if user_id:
            query = query.filter_by(user_id=user_id)
            
        # 按时间降序排序
        records = query.order_by(DetectionRecord.created_at.desc()).all()
        
        # 获取用户信息
        users = {user.id: user.username for user in User.query.all()}
        
        return jsonify([{
            'id': record.id,
            'user_id': record.user_id,
            'username': users.get(record.user_id, '未知用户'),
            'created_at': record.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'visible_image_path': record.visible_image_path,
            'infrared_image_path': record.infrared_image_path,
            'result_image_path': record.result_image_path,
            'detection_details': record.detection_details,
            'image_sequence': record.image_sequence  # 添加图片序号
        } for record in records])
    except Exception as e:
        logger.error(f"获取检测记录失败: {str(e)}")
        return jsonify({'error': '获取检测记录失败'}), 500

@app.route('/api/admin/add_user', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '没有收到数据'}), 400
            
        username = data.get('username')
        password = data.get('password')
        is_admin = data.get('is_admin', False)
        
        if not all([username, password]):
            return jsonify({'error': '用户名和密码不能为空'}), 400
            
        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            return jsonify({'error': '用户名已存在'}), 400
            
        # 创建新用户
        new_user = User(
            username=username,
            password_hash=generate_password_hash(password),
            is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({'message': '用户添加成功'})
        
    except Exception as e:
        logger.error(f"添加用户失败: {str(e)}")
        db.session.rollback()
        return jsonify({'error': '添加用户失败'}), 500

@app.route('/api/admin/reset_password', methods=['POST'])
def reset_user_password():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '没有收到数据'}), 400
            
        user_id = data.get('user_id')
        new_password = data.get('new_password')
        
        if not all([user_id, new_password]):
            return jsonify({'error': '参数不完整'}), 400
            
        # 查找用户
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        # 更新密码
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        return jsonify({'message': '密码重置成功'})
        
    except Exception as e:
        logger.error(f"重置密码失败: {str(e)}")
        db.session.rollback()
        return jsonify({'error': '重置密码失败'}), 500

@app.route('/api/admin/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '没有收到数据'}), 400
            
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({'error': '用户ID不能为空'}), 400
            
        # 查找用户
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': '用户不存在'}), 404
            
        # 删除用户
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': '用户删除成功'})
        
    except Exception as e:
        logger.error(f"删除用户失败: {str(e)}")
        db.session.rollback()
        return jsonify({'error': '删除用户失败'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 