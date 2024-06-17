import '../index.css';

const TaskAssignments = () => {
    return (
        <div className="bg-gray-100 min-h-screen flex">
            {/* Sidebar */}
            <div className="w-60 bg-white shadow-md px-5 py-7">
                <h1 className="text-xl font-semibold mb-10">Optimization</h1>
                <ul className="space-y-4">
                <li><a href="#" className="text-gray-700 hover:text-gray-900">Toolbox</a></li>
                <li><a href="#" className="text-gray-700 hover:text-gray-900">Results</a></li>
                </ul>
            </div>

            {/* Main Content */}
            <div className="flex-grow p-10">
                <div className="max-w-4xl mx-auto">
                <h2 className="text-2xl font-bold mb-6">Task Assignment</h2>
                <div className="flex gap-10">
                    {/* Parameter section */}
                    <div className="bg-white shadow-lg p-6 flex-1 rounded">
                    <h3 className="text-lg font-semibold mb-4">Overview</h3>
                    <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                        <label className="flex-1">Overqualification:</label>
                        <button className="bg-gray-200 hover:bg-gray-300 px-2 py-1 rounded">True</button>
                        <button className="bg-gray-200 hover:bg-gray-300 px-2 py-1 rounded">False</button>
                        </div>
                        <div>
                        <label>Max Employee Workload</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>Weight Idle</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>Weight Score</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>Weight Balance</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>Presolve</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>Heuristics</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>MIPFocus</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>MIPGap</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                        <div>
                        <label>Threads</label>
                        <input type="number" className="border-gray-300 rounded w-full" />
                        </div>
                    </div>
                    </div>

                    {/* File upload section */}
                    <div className="bg-white shadow-lg p-6 flex-1 rounded">
                    <h3 className="text-lg font-semibold mb-4">Upload Data</h3>
                    <div className="space-y-4">
                        <div>
                        <label>Upload Employee Data</label>
                        <button className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Upload</button>
                        </div>
                        <div>
                        <label>Upload Task Data</label>
                        <button className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Upload</button>
                        </div>
                        <div>
                        <label>Upload Licensed File</label>
                        <button className="w-full py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">Upload</button>
                        </div>
                    </div>
                    </div>
                </div>
                </div>
            </div>
        </div>
    );
}

export default TaskAssignments;
