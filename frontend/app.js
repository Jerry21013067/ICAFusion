const { createApp } = Vue

createApp({
    data() {
        return {
            isLoggedIn: false,
            currentUser: null,
            showingUserCenter: false,
            showingHome: false,
            userCenterTab: 'info',
            adminCenterTab: 'info',
            loginForm: {
                username: '',
                password: ''
            },
            registerForm: {
                username: '',
                password: ''
            },
            passwordForm: {
                oldPassword: '',
                newPassword: '',
                confirmPassword: ''
            },
            visibleImage: null,
            infraredImage: null,
            visiblePreview: null,
            infraredPreview: null,
            resultImage: null,
            detectionDetails: [],
            detectionRecords: [],
            loginModal: null,
            registerModal: null,
            isProcessing: false,
            apiBaseUrl: 'http://localhost:5000',
            // 分页相关
            currentPage: 1,
            pageSize: 10,
            totalRecords: 0,
            sortOrder: 'desc', // 'desc' 或 'asc'
            pageSizeOptions: [5, 10, 20, 30],
            // 管理员中心分页
            adminCurrentPage: 1,
            adminPageSize: 10,
            adminTotalRecords: 0,
            // 用户列表分页
            userCurrentPage: 1,
            userPageSize: 10,
            userTotalRecords: 0,
            currentDetectionDetails: [],  // 当前显示的检测详情
            detectionDetailsModal: null,  // 检测详情模态框实例
            userSearchQuery: '',
            allUsers: [],
            allRecords: [],
            newUser: {
                username: '',
                password: '',
                is_admin: false
            },
            resetPasswordForm: {
                userId: null,
                newPassword: ''
            },
            resetPasswordModal: null,
            selectedUserId: '', // 添加选中的用户ID
            imageSequenceQuery: '', // 添加图片序号搜索关键词
        }
    },
    computed: {
        canSubmit() {
            return this.visibleImage && this.infraredImage && !this.isProcessing
        },
        totalPages() {
            return Math.ceil(this.totalRecords / this.pageSize)
        },
        adminTotalPages() {
            return Math.ceil(this.adminTotalRecords / this.adminPageSize)
        },
        userTotalPages() {
            return Math.ceil(this.userTotalRecords / this.userPageSize)
        },
        paginatedRecords() {
            let records = this.detectionRecords;
            
            // 添加图片序号筛选
            if (this.imageSequenceQuery) {
                const query = this.imageSequenceQuery.toLowerCase();
                records = records.filter(record => 
                    record.image_sequence && record.image_sequence.toLowerCase().includes(query)
                );
            }
            
            const start = (this.currentPage - 1) * this.pageSize;
            const end = start + this.pageSize;
            return records.slice(start, end);
        },
        paginatedFilteredRecords() {
            let records = this.filteredRecords;
            const start = (this.adminCurrentPage - 1) * this.adminPageSize;
            const end = start + this.adminPageSize;
            return records.slice(start, end);
        },
        paginatedFilteredUsers() {
            let users = this.filteredUsers;
            const start = (this.userCurrentPage - 1) * this.userPageSize;
            const end = start + this.userPageSize;
            return users.slice(start, end);
        },
        filteredUsers() {
            if (!this.userSearchQuery) return this.allUsers;
            return this.allUsers.filter(user => 
                user.username.toLowerCase().includes(this.userSearchQuery.toLowerCase())
            );
        },
        filteredRecords() {
            let records = this.allRecords;
            
            // 先按用户ID筛选
            if (this.selectedUserId) {
                records = records.filter(record => record.user_id === this.selectedUserId);
            }
            
            // 再按用户名搜索关键词筛选
            if (this.userSearchQuery) {
                const query = this.userSearchQuery.toLowerCase();
                records = records.filter(record => 
                    record.username.toLowerCase().includes(query)
                );
            }
            
            // 最后按图片序号筛选
            if (this.imageSequenceQuery) {
                const query = this.imageSequenceQuery.toLowerCase();
                records = records.filter(record => 
                    record.image_sequence && record.image_sequence.toLowerCase().includes(query)
                );
            }
            
            return records;
        }
    },
    mounted() {
        // 初始化模态框
        this.loginModal = new bootstrap.Modal(document.getElementById('loginModal'))
        this.registerModal = new bootstrap.Modal(document.getElementById('registerModal'))
        this.resetPasswordModal = new bootstrap.Modal(document.getElementById('resetPasswordModal'))
        
        // 检查本地存储中的登录状态
        const userId = localStorage.getItem('userId')
        const username = localStorage.getItem('username')
        const isAdmin = localStorage.getItem('isAdmin') === 'true'  // 确保正确解析布尔值
        
        console.log('恢复登录状态:', { userId, username, isAdmin });  // 添加日志
        
        if (userId) {
            this.isLoggedIn = true
            this.currentUser = {
                id: userId,
                username: username,
                is_admin: isAdmin
            }
            console.log('设置当前用户信息:', this.currentUser);  // 添加日志
            
            // 根据用户角色决定显示界面
            if (isAdmin) {
                console.log('管理员恢复登录状态，显示管理员中心');
                this.showUserCenter();
            } else {
                console.log('普通用户恢复登录状态，显示主页');
                this.showHome();
            }
        }
        
        // 初始化检测详情模态框
        this.$nextTick(() => {
            const modalElement = document.getElementById('detectionDetailsModal')
            if (modalElement) {
                this.detectionDetailsModal = new bootstrap.Modal(modalElement)
            } else {
                console.error('检测详情模态框元素未找到')
            }
        })
    },
    methods: {
        showUserCenter() {
            console.log('显示个人中心');
            this.showingHome = false;
            this.showingUserCenter = true;
            this.userCenterTab = 'info';
            this.adminCenterTab = 'info';
            // 如果是管理员，加载管理员需要的数据
            if (this.currentUser && this.currentUser.is_admin) {
                this.loadAllUsers();
                this.loadAllRecords();
            } else {
                // 如果是普通用户，加载检测记录
                this.loadDetectionRecords();
            }
        },
        showAdminCenter() {
            console.log('显示管理员中心');
            this.showingHome = false;
            this.showingUserCenter = true;
            this.userCenterTab = 'info';
            this.adminCenterTab = 'info';
            // 加载管理员需要的数据
            this.loadAllUsers();
            this.loadAllRecords();
        },
        showRecords() {
            console.log('显示检测记录');
            this.showingHome = false;
            this.showingUserCenter = true;
            this.userCenterTab = 'records';
            this.loadDetectionRecords();
        },
        showHome() {
            console.log('显示首页');
            this.showingHome = true;
            this.showingUserCenter = false;
        },
        showLoginModal() {
            this.loginModal.show()
        },
        showRegisterModal() {
            this.registerModal.show()
        },
        hideUserCenter() {
            this.showingUserCenter = false;
        },
        async loadDetectionRecords() {
            try {
                const response = await fetch(`/api/detection_records?user_id=${this.currentUser.id}&sort_order=${this.sortOrder}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log('检测记录数据:', data);
                
                if (Array.isArray(data)) {
                    this.detectionRecords = data;
                    this.totalRecords = data.length;
                } else {
                    console.error('返回的数据不是数组:', data);
                    this.detectionRecords = [];
                    this.totalRecords = 0;
                }
                
                // 重置到第一页
                this.currentPage = 1;
            } catch (error) {
                console.error('加载检测记录失败:', error);
                this.detectionRecords = [];
                this.totalRecords = 0;
                alert('加载检测记录失败，请稍后重试');
            }
        },
        formatDate(dateString) {
            const date = new Date(dateString);
            return date.toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        },
        openImage(path) {
            window.open('/' + path, '_blank');
        },
        async updatePassword() {
            if (this.passwordForm.newPassword !== this.passwordForm.confirmPassword) {
                alert('两次输入的新密码不一致');
                return;
            }
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/update_password?user_id=${this.currentUser.id}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        old_password: this.passwordForm.oldPassword,
                        new_password: this.passwordForm.newPassword
                    })
                });
                if (response.ok) {
                    alert('密码修改成功');
                    this.passwordForm = {
                        oldPassword: '',
                        newPassword: '',
                        confirmPassword: ''
                    };
                } else {
                    const data = await response.json();
                    alert(data.error || '密码修改失败');
                }
            } catch (error) {
                console.error('密码修改失败:', error);
                alert('密码修改失败');
            }
        },
        async login() {
            try {
                console.log('开始登录请求:', this.loginForm);
                const response = await fetch(`${this.apiBaseUrl}/api/login`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.loginForm)
                });
                
                const data = await response.json();
                console.log('登录响应数据:', data);  // 添加详细日志
                console.log('登录响应数据类型:', typeof data);  // 添加类型检查
                console.log('登录响应数据的所有字段:', Object.keys(data));  // 添加字段检查
                
                if (response.ok) {
                    this.isLoggedIn = true;
                    // 修改判断逻辑，确保正确处理数字1
                    const isAdmin = data.is_admin === true || data.is_admin === 1 || data.is_admin === '1';
                    console.log('is_admin 原始值:', data.is_admin);
                    console.log('is_admin 处理后:', isAdmin);
                    
                    this.currentUser = {
                        id: data.user_id,
                        username: data.username,
                        is_admin: isAdmin
                    };
                    console.log('设置当前用户信息:', this.currentUser);
                    
                    localStorage.setItem('userId', data.user_id);
                    localStorage.setItem('username', data.username);
                    localStorage.setItem('isAdmin', isAdmin);
                    
                    this.loginModal.hide();
                    this.loginForm = { username: '', password: '' };
                    
                    // 根据用户角色决定显示界面
                    if (isAdmin) {
                        console.log('管理员登录，显示管理员中心');
                        this.showUserCenter();
                    } else {
                        console.log('普通用户登录，显示主页');
                        this.showHome();
                    }
                } else {
                    alert(data.error || '登录失败，请重试');
                }
            } catch (error) {
                console.error('登录请求失败:', error);
                alert('登录失败，请检查网络连接或稍后重试');
            }
        },
        async register() {
            try {
                console.log('开始注册请求:', this.registerForm);
                const response = await fetch(`${this.apiBaseUrl}/api/register`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.registerForm)
                });
                
                const data = await response.json();
                console.log('注册响应:', data);
                
                if (response.ok) {
                    alert('注册成功，请登录');
                    this.registerModal.hide();
                    this.showLoginModal();
                    this.registerForm = { username: '', password: '' };
                } else {
                    alert(data.error || '注册失败，请重试');
                }
            } catch (error) {
                console.error('注册请求失败:', error);
                alert('注册失败，请检查网络连接或稍后重试');
            }
        },
        logout() {
            this.isLoggedIn = false;
            this.currentUser = null;
            this.showingUserCenter = false;
            this.userCenterTab = 'info';
            this.adminCenterTab = 'info';
            localStorage.removeItem('userId');
            localStorage.removeItem('username');
            localStorage.removeItem('isAdmin');
            this.visibleImage = null;
            this.infraredImage = null;
            this.visiblePreview = null;
            this.infraredPreview = null;
            this.resultImage = null;
            this.detectionDetails = [];
            this.detectionRecords = [];
            this.allUsers = [];
            this.allRecords = [];
            this.userSearchQuery = '';
            this.newUser = {
                username: '',
                password: '',
                is_admin: false
            };
            this.resetPasswordForm = {
                userId: null,
                newPassword: ''
            };
        },
        handleVisibleImage(event) {
            const file = event.target.files[0]
            if (file) {
                this.visibleImage = file
                this.createPreview(file, 'visible')
                console.log('可见光图片预览已创建')
            }
        },
        handleInfraredImage(event) {
            const file = event.target.files[0]
            if (file) {
                this.infraredImage = file
                this.createPreview(file, 'infrared')
                console.log('红外图片预览已创建')
            }
        },
        createPreview(file, type) {
            const reader = new FileReader()
            reader.onload = e => {
                if (type === 'visible') {
                    this.visiblePreview = e.target.result
                    console.log('可见光预览URL:', this.visiblePreview)
                } else {
                    this.infraredPreview = e.target.result
                    console.log('红外预览URL:', this.infraredPreview)
                }
            }
            reader.readAsDataURL(file)
        },
        async submitImages() {
            if (!this.canSubmit) return
            
            this.isProcessing = true
            const formData = new FormData()
            formData.append('visible_image', this.visibleImage)
            formData.append('infrared_image', this.infraredImage)
            formData.append('user_id', this.currentUser.id)
            
            try {
                console.log('开始发送检测请求')
                const response = await fetch(`${this.apiBaseUrl}/api/detect`, {
                    method: 'POST',
                    body: formData
                })
                
                const data = await response.json()
                console.log('检测响应:', data)
                
                if (response.ok) {
                    // 确保使用完整的URL
                    const resultUrl = `${this.apiBaseUrl}/${data.result_image}`
                    console.log('结果图片URL:', resultUrl)
                    this.resultImage = resultUrl
                    this.detectionDetails = data.detection_details || []
                } else {
                    alert(data.error || '检测失败，请重试')
                }
            } catch (error) {
                console.error('检测失败:', error)
                alert('检测失败，请重试')
            } finally {
                this.isProcessing = false
            }
        },
        changePage(page) {
            if (page >= 1 && page <= this.totalPages) {
                this.currentPage = page;
            }
        },
        changeAdminPage(page) {
            if (page >= 1 && page <= this.adminTotalPages) {
                this.adminCurrentPage = page;
            }
        },
        changeUserPage(page) {
            if (page >= 1 && page <= this.userTotalPages) {
                this.userCurrentPage = page;
            }
        },
        changePageSize(size) {
            this.pageSize = parseInt(size);
            this.currentPage = 1;
            this.loadDetectionRecords();
        },
        changeAdminPageSize(size) {
            this.adminPageSize = parseInt(size);
            this.adminCurrentPage = 1;
            this.loadAllRecords();
        },
        changeUserPageSize(size) {
            this.userPageSize = parseInt(size);
            this.userCurrentPage = 1;
            this.loadAllUsers();
        },
        toggleSortOrder() {
            this.sortOrder = this.sortOrder === 'desc' ? 'asc' : 'desc';
            this.loadDetectionRecords();
        },
        showDetectionDetails(details) {
            this.currentDetectionDetails = details;
            this.detectionDetailsModal.show();
        },
        // 监听标签页切换
        watch: {
            userCenterTab(newTab) {
                console.log('个人中心标签页切换:', newTab);
                if (newTab === 'records') {
                    this.loadDetectionRecords();
                }
            },
            adminCenterTab(newTab) {
                console.log('管理员中心标签页切换:', newTab);
                if (newTab === 'users') {
                    this.loadAllUsers();
                } else if (newTab === 'records') {
                    this.loadAllRecords();
                }
            }
        },
        loadUserData() {
            if (this.currentUser && this.currentUser.is_admin) {
                this.loadAllUsers();
                this.loadAllRecords();
            } else {
                this.loadDetectionRecords();
            }
        },
        async loadAllUsers() {
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/admin/users`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                this.allUsers = data;
                this.userTotalRecords = data.length;
                // 重置到第一页
                this.userCurrentPage = 1;
            } catch (error) {
                console.error('加载用户列表失败:', error);
                alert('加载用户列表失败');
                this.allUsers = [];
                this.userTotalRecords = 0;
            }
        },
        async loadAllRecords() {
            try {
                console.log('开始加载检测记录，选中的用户ID:', this.selectedUserId);
                const url = `${this.apiBaseUrl}/api/admin/all_records${this.selectedUserId ? `?user_id=${this.selectedUserId}` : ''}`;
                console.log('请求URL:', url);
                
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log('获取到的检测记录:', data);
                
                if (Array.isArray(data)) {
                    this.allRecords = data;
                    this.adminTotalRecords = data.length;
                    // 重置到第一页
                    this.adminCurrentPage = 1;
                } else {
                    console.error('返回的数据不是数组:', data);
                    this.allRecords = [];
                    this.adminTotalRecords = 0;
                }
            } catch (error) {
                console.error('加载检测记录失败:', error);
                alert('加载检测记录失败: ' + error.message);
                this.allRecords = [];
                this.adminTotalRecords = 0;
            }
        },
        showAddUserModal() {
            this.newUser = {
                username: '',
                password: '',
                is_admin: false
            };
            const modal = new bootstrap.Modal(document.getElementById('addUserModal'));
            modal.show();
        },
        async addUser() {
            try {
                if (!this.newUser.username || !this.newUser.password) {
                    alert('用户名和密码不能为空');
                    return;
                }
                
                console.log('准备添加用户:', this.newUser);
                const response = await fetch(`${this.apiBaseUrl}/api/admin/add_user`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        username: this.newUser.username,
                        password: this.newUser.password,
                        is_admin: this.newUser.is_admin
                    })
                });
                
                const data = await response.json();
                console.log('添加用户响应:', data);
                
                if (response.ok) {
                    alert('添加用户成功');
                    this.loadAllUsers();
                    const modal = bootstrap.Modal.getInstance(document.getElementById('addUserModal'));
                    modal.hide();
                    // 重置表单
                    this.newUser = {
                        username: '',
                        password: '',
                        is_admin: false
                    };
                } else {
                    alert(data.error || '添加用户失败');
                }
            } catch (error) {
                console.error('添加用户失败:', error);
                alert('添加用户失败');
            }
        },
        showResetPasswordModal(user) {
            this.resetPasswordForm = {
                userId: user.id,
                newPassword: ''
            };
            this.resetPasswordModal.show();
        },
        async resetPassword() {
            try {
                if (!this.resetPasswordForm.newPassword) {
                    alert('新密码不能为空');
                    return;
                }
                
                console.log('准备重置密码:', this.resetPasswordForm);
                const response = await fetch(`${this.apiBaseUrl}/api/admin/reset_password`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        user_id: this.resetPasswordForm.userId,
                        new_password: this.resetPasswordForm.newPassword
                    })
                });
                
                const data = await response.json();
                console.log('重置密码响应:', data);
                
                if (response.ok) {
                    alert('重置密码成功');
                    this.resetPasswordModal.hide();
                    // 重置表单
                    this.resetPasswordForm = {
                        userId: null,
                        newPassword: ''
                    };
                } else {
                    alert(data.error || '重置密码失败');
                }
            } catch (error) {
                console.error('重置密码失败:', error);
                alert('重置密码失败');
            }
        },
        async deleteUser(user) {
            if (confirm('确定要删除该用户吗？此操作不可恢复。')) {
                try {
                    const response = await fetch(`${this.apiBaseUrl}/api/admin/delete_user`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ user_id: user.id })
                    });
                    
                    if (response.ok) {
                        alert('删除用户成功');
                        this.loadAllUsers();
                    } else {
                        const data = await response.json();
                        alert(data.error || '删除用户失败');
                    }
                } catch (error) {
                    console.error('删除用户失败:', error);
                    alert('删除用户失败');
                }
            }
        },
        handleUserFilter() {
            // 不再需要重新加载数据，因为使用计算属性实时筛选
            console.log('筛选条件更新:', {
                selectedUserId: this.selectedUserId,
                searchQuery: this.userSearchQuery
            });
        },
    }
}).mount('#app') 