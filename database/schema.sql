-- Supabase PostgreSQL Schema for EduPulse AI

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Profiles Table (Extends Supabase Auth)
CREATE TABLE public.profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    full_name TEXT,
    role TEXT CHECK (role IN ('Administrator', 'Teacher', 'Academic Counselor', 'Viewer')),
    department TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Students Table
CREATE TABLE public.students (
    student_id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER,
    gender TEXT,
    department TEXT,
    semester INTEGER,
    enrollment_date DATE DEFAULT CURRENT_DATE,
    status TEXT DEFAULT 'Active' CHECK (status IN ('Active', 'Archived', 'Graduated')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Academic Records Table
CREATE TABLE public.academic_records (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    student_id TEXT REFERENCES public.students(student_id) ON DELETE CASCADE,
    study_hours_per_week NUMERIC,
    attendance_percentage NUMERIC,
    assignment_average NUMERIC,
    midterm_score NUMERIC,
    previous_gpa NUMERIC,
    internet_access BOOLEAN,
    extra_academic_support BOOLEAN,
    part_time_job BOOLEAN,
    extracurricular_hours_per_week NUMERIC,
    absences INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. Predictions Table
CREATE TABLE public.predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    student_id TEXT REFERENCES public.students(student_id) ON DELETE CASCADE,
    input_snapshot JSONB NOT NULL,
    prediction TEXT NOT NULL,
    probability NUMERIC NOT NULL,
    risk_level TEXT NOT NULL,
    model_version TEXT,
    user_id UUID REFERENCES auth.users(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 5. Model Versions Table
CREATE TABLE public.model_versions (
    version TEXT PRIMARY KEY,
    description TEXT,
    f1_score NUMERIC,
    deployed_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    is_active BOOLEAN DEFAULT FALSE
);

-- 6. Audit Logs Table
CREATE TABLE public.audit_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Indexes for performance
CREATE INDEX idx_students_department ON public.students(department);
CREATE INDEX idx_predictions_student_id ON public.predictions(student_id);
CREATE INDEX idx_predictions_timestamp ON public.predictions(timestamp);
CREATE INDEX idx_academic_records_student_id ON public.academic_records(student_id);

-- Row Level Security (RLS) Policies
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.students ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.academic_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to view profiles
CREATE POLICY "Allow authenticated users to view profiles" ON public.profiles FOR SELECT USING (auth.role() = 'authenticated');

-- Allow users to update their own profile
CREATE POLICY "Allow users to update own profile" ON public.profiles FOR UPDATE USING (auth.uid() = id);

-- Allow authenticated users to view students
CREATE POLICY "Allow authenticated users to view students" ON public.students FOR SELECT USING (auth.role() = 'authenticated');

-- Allow admin/teachers/counselors to insert/update students
CREATE POLICY "Allow privileged users to modify students" ON public.students FOR ALL USING (
    EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('Administrator', 'Teacher', 'Academic Counselor'))
);

-- Allow authenticated users to view predictions
CREATE POLICY "Allow authenticated users to view predictions" ON public.predictions FOR SELECT USING (auth.role() = 'authenticated');

-- Allow privileged users to insert predictions
CREATE POLICY "Allow privileged users to insert predictions" ON public.predictions FOR INSERT WITH CHECK (
    EXISTS (SELECT 1 FROM public.profiles WHERE id = auth.uid() AND role IN ('Administrator', 'Teacher', 'Academic Counselor'))
);
